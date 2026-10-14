from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from transformers import PreTrainedTokenizer

from agent_system.agent.agents.base import BaseAgent
from agent_system.agent.registry import AgentRegistry
from agent_system.environments.env_package.openresearcher.action import encode_action
from agent_system.environments.env_package.openresearcher.context import fit_messages, resolve_dialect
from agent_system.environments.env_package.openresearcher.parser import normalize_native_markup, cut_at_stop, parse_assistant_turn
from agent_system.multi_turn_rollout.utils import preprocess_batch
from agent_system.agent.agents.openresearcher.response_mask import trim_response_mask
from verl import DataProto

ACT_STOP = ["<tool_response>"]     # subsumes OR's "\n<tool_response>" for stopping purposes


def _or_cfg(config):
    return getattr(getattr(config, "env", None), "openresearcher", None)


def _cfg_get(cfg, k, d):
    if cfg is None:
        return d
    try:
        v = cfg.get(k, d)
    except AttributeError:
        v = getattr(cfg, k, d)
    return d if v is None else v


class _OrAgentBase(BaseAgent):
    """Shared prompt machinery for the chain agents."""
    AGENT_NAME = "Researcher Agent (openresearcher)"
    SYSTEM_SUFFIX = ""             # agents append their role here

    def __init__(self, wg_id: str, tokenizer: PreTrainedTokenizer, processor, config: Any):
        ocfg = _or_cfg(config)
        self._margin = int(_cfg_get(ocfg, "prompt_margin", 64))
        self._keep_last_k = int(_cfg_get(ocfg, "keep_last_k", 2))
        try:
            self._budget = int(config.data.max_prompt_length) - self._margin
        except Exception:
            self._budget = 16384 - self._margin
        # native (Qwen3 renders tools= + the tool role) | textual (gemma: tools folded into the
        # system text, tool results as <tool_response> user turns) | auto. preprocess_fn resolves
        # the same key against the same tokenizer, so the budget is measured on the bytes that are
        # actually generated from; the SFT data was rendered through the same path.
        self._dialect = resolve_dialect(tokenizer, str(_cfg_get(ocfg, "tool_dialect", "auto")))
        # stub | header: what an elided tool page becomes (must equal the SFT data's setting)
        self._elide_mode = str(_cfg_get(ocfg, "elide_mode", "stub"))
        # `prompt` is unused (the env supplies the messages) but BaseAgent requires a str
        super().__init__(self.AGENT_NAME, "{env_prompt}", wg_id=wg_id, tokenizer=tokenizer,
                         processor=None, config=config)

    # ---- views -------------------------------------------------------------------------------
    def _view(self, messages: List[Dict[str, Any]], trailing_user: str = "") -> List[Dict[str, Any]]:
        msgs = [dict(m) for m in messages]
        if self.SYSTEM_SUFFIX and msgs and msgs[0].get("role") == "system":
            msgs[0]["content"] = msgs[0]["content"] + "\n\n" + self.SYSTEM_SUFFIX
        if trailing_user:
            msgs.append({"role": "user", "content": trailing_user})
        if self.tokenizer is None:            # scripted tests: no budget check possible
            return msgs
        msgs, _ = fit_messages(msgs, self._tools(), self.tokenizer, self._budget, self._keep_last_k,
                               dialect=self._dialect, elide_mode=self._elide_mode)
        return msgs

    @staticmethod
    def _tools_of(env_obs):
        return env_obs.get("tools")

    def _tools(self):
        return getattr(self, "_current_tools", None)

    def _build_obs(self, env_obs: Dict[str, Any], trailing_user: str = "") -> Dict[str, Any]:
        self._current_tools = self._tools_of(env_obs)
        obs = {k: v for k, v in env_obs.items() if k != "messages"}
        obs["image"] = None
        # the tool list the prompt is rendered with (preprocess_fn reads obs["tools"]): an agent
        # sees only its own slice (_tools_of)
        obs["tools"] = self._current_tools
        obs["messages"] = [self._view(m, trailing_user) for m in env_obs["messages"]]
        return obs

    # trailing USER turn of the act prompt; each agent sets its "your turn" instruction. The SFT
    # act rows were rendered through act_obs too.
    ACT_TRAILING = ""

    def act_obs(self, env_obs: Dict[str, Any]) -> Dict[str, Any]:
        return self._build_obs(env_obs, trailing_user=self.ACT_TRAILING)

    def build_prompt(self, env_obs: Dict[str, Any], team_context: List[str], step: int):
        return self.act_obs(env_obs)

    # ---- diagnostics -------------------------------------------------------------------------
    _NATIVE_MARKUP = ("<|channel>", "<|tool_call>", "<|tool_response>")

    def _texts_and_caps(self, batch: DataProto, text_responses):
        """(texts, hit_cap) per row. BaseAgent decodes with skip_special_tokens=True, which silently drops
        gemma's native markup (a `<tool_call|>` closing a textual `<tool_call>` block became an unclosed
        tag = a "truncated" episode). Re-decode WITH the special tokens, map them to the textual tags
        (parser.normalize_native_markup) and record whether the row actually filled the response budget
        (no padding, last token not a terminal special token) -- the env's truncation rule uses that flag."""
        try:
            ids = batch.batch["responses"]
            pad = self.tokenizer.pad_token_id
            terminal = set(int(i) for i in (self.tokenizer.all_special_ids or []))
            texts, caps = [], []
            for i in range(ids.shape[0]):
                row = ids[i]
                keep = row if pad is None else row[row != pad]
                texts.append(normalize_native_markup(self.tokenizer.decode(keep, skip_special_tokens=False)))
                caps.append(bool(len(keep) == ids.shape[1] and int(row[-1]) not in terminal))
            return texts, caps
        except Exception:  # noqa: BLE001 -- fall back to the shared decode; the env keeps its textual heuristic
            return [t if isinstance(t, str) else "" for t in text_responses], [None] * len(text_responses)

    def _report_native_markup(self, batch: DataProto, agent_active_mask, kind: str) -> None:
        """BaseAgent decodes with skip_special_tokens=True, so gemma's own thinking / tool-call
        tokens vanish from the text the env and the parser see (observed: a correct answer read
        "thought\nThinking Process: ..." -- the base model's <|channel>thought leaking through the
        SFT'd <think> convention). Count them here from the ids and say so;
        the text path is left as it is (OR-only; BaseAgent is shared with the other benches)."""
        try:
            ids = batch.batch['responses']
            special = [self.tokenizer.convert_tokens_to_ids(t) for t in self._NATIVE_MARKUP]
            special = [i for i in special if isinstance(i, int) and i >= 0 and i != getattr(self.tokenizer, "unk_token_id", -1)]
            if not special:
                return
            import torch
            hit = torch.zeros(ids.shape[0], dtype=torch.bool)
            for i in special:
                hit |= (ids == i).any(dim=-1)
            mask = np.asarray(agent_active_mask).astype(bool)
            k, n = int(hit.numpy()[mask].sum()), int(mask.sum())
            if k:
                print(f"[openresearcher] {self.AGENT_NAME} {kind}: native gemma markup ({'/'.join(self._NATIVE_MARKUP)}) in {k}/{n} responses -- invisible to the parser (skip_special_tokens)", flush=True)
        except Exception:  # noqa: BLE001 -- diagnostics never kill a rollout
            pass

    # ---- the act call ------------------------------------------------------------------------
    def call(self, gen_batch: DataProto, env_obs: Dict[str, Any], team_context: List[str],
             actor_rollout_wg, agent_active_mask, step: int) -> Tuple[DataProto, List[str]]:
        obs = self.build_prompt(env_obs, team_context, step)
        batch = preprocess_batch(gen_batch=gen_batch, obs=obs, config=self.config,
                                 tokenizer=self.tokenizer, processor=self.processor)
        meta = {**gen_batch.meta_info, 'sampling_overrides': {'stop': list(ACT_STOP)}}
        batch, text_responses = self._generate_with_llm(batch, actor_rollout_wg, agent_active_mask,
                                                        meta, kind="act")
        trim_response_mask(batch, self.tokenizer.pad_token_id, f"{self.name}/act")
        self._report_native_markup(batch, agent_active_mask, "act")
        text_responses, caps = self._texts_and_caps(batch, text_responses)
        emitted, valids = [], []
        for i, t in enumerate(text_responses):
            txt = cut_at_stop(t if isinstance(t, str) else "")
            emitted.append(encode_action(txt, None, caps[i]) if caps[i] is not None else txt)
            valids.append(bool(agent_active_mask[i]) and not parse_assistant_turn(txt).parse_error)
        n = len(text_responses)
        batch.non_tensor_batch['is_action_valid'] = np.array(valids, dtype=bool)
        batch.non_tensor_batch['env_step'] = np.array([step] * n, dtype=object)
        batch.non_tensor_batch['call_kind'] = np.array(['act'] * n, dtype=object)
        return batch, emitted
