from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import numpy as np
from transformers import PreTrainedTokenizer

from verl import DataProto
from agent_system.agent.agents.base import BaseAgent
from agent_system.agent.registry import AgentRegistry
from agent_system.agent.utils import parse_open_bid
from agent_system.multi_turn_rollout.utils import preprocess_batch

# NaVILA's question, verbatim except the trailing agent role. The 8 <image> markers are added by the
# base multimodal preprocess (is_navila branch expands them); we keep the exact NAVILA_PROMPT body
# so instruction-following stays in-distribution.
_NAVILA_Q = (
    "Imagine you are a robot programmed for navigation tasks. You have been given a video of "
    "historical observations <image>\n<image>\n<image>\n<image>\n<image>\n<image>\n<image>\n, and "
    "current observation <image>\n. Your assigned task is: \"{env_prompt}\" "
)

BID_BODY = (
    _NAVILA_Q +
    "You are the {desk} desk, one of {n_desks} specialists (Turner/Mover/Stopper) that share control "
    "of this robot. Exactly ONE desk acts each step, decided by a sealed-bid auction you are bidding "
    "in now. If you win you may ONLY emit: {verbs}. You cannot emit anything else, so bid only for "
    "steps your action can advance.\n"
    "# Bidding (revenue share, five levels)\n"
    "Your bid is the share of your income this step you pass back to the desk that acted before you. "
    "The HIGHEST bid wins; only the winner pays a small rent. Read the frames and the instruction, "
    "and bid:\n"
    "  0    = pass. My action cannot help now; another desk must act first.\n"
    "  0.2  = my action barely helps here.\n"
    "  0.4  = my action is useful here.\n"
    "  0.6  = my action is important here.\n"
    "  0.8  = I am the bottleneck: the robot cannot progress without my action right now.\n"
    "Your bid MUST be exactly one of: 0, 0.2, 0.4, 0.6, 0.8. Answer with ONE short sentence of "
    "reasoning about the scene, then the number:\n"
    "<think>one sentence</think>\n<bid>0.4</bid>\n"
)

ACT_BODY = (
    _NAVILA_Q +
    "You are the {desk} desk and you WON this step's auction. You issue the single navigation action "
    "for this step. HARD CONSTRAINT, enforced by a parser: your action must be {verbs}. Anything else "
    "is REJECTED, the step is wasted, and you still pay. {desk_hint}\n"
    "Analyze the images to decide your action. "
)

# The model emits <think>...</think>\n<act>ACTION</act>; the env parses the ACTION inside <act>.
_ACT_TAG = re.compile(r"<act>(.*?)</act>", re.IGNORECASE | re.DOTALL)

# Per-agent whitelist: an emitted action is in-slice iff its <act> verb matches this regex.
_TURN_RE = re.compile(r"\bturn\s+(left|right)\b", re.I)
_MOVE_RE = re.compile(r"\bmove\s+forward\b", re.I)
_STOP_RE = re.compile(r"\bstop\b", re.I)


class _NavilaDeskBase(BaseAgent):
    AGENT_NAME = "Desk"
    DESK = "Desk"
    VERBS = ""            # human-readable slice description for the prompt
    SLICE_RE = None       # compiled regex the emitted action must match to be in-slice
    HINT = ""
    SEES_IMAGE = True

    def __init__(self, wg_id: str, tokenizer: PreTrainedTokenizer, processor, config: Any):
        cfg = getattr(getattr(getattr(config, "agent", None), "orchestra", None), "navila_chain", None)

        def _get(k, d):
            return getattr(cfg, k, d) if cfg is not None else d

        self._bid_max = float(_get("bid_max", 1.0))
        self._bid_max_tokens = int(_get("bid_max_tokens", 64))
        self._bid_image = bool(_get("bid_sees_image", True))
        try:
            self._n_desks = len(config.agent.agent_ids)
        except Exception:
            self._n_desks = 3

        def _fill(body: str) -> str:
            for k, v in (("{desk}", self.DESK), ("{verbs}", self.VERBS),
                         ("{desk_hint}", self.HINT), ("{n_desks}", str(self._n_desks))):
                body = body.replace(k, v)
            return body

        self._bid_prompt = _fill(BID_BODY)
        act_prompt = _fill(ACT_BODY)
        super().__init__(self.AGENT_NAME, act_prompt, wg_id=wg_id, tokenizer=tokenizer,
                         processor=processor, config=config)

    # -- prompt build (8-frame window kept via env_obs['image'] = list of frames) --------------
    def _build_obs(self, template: str, env_obs, team_context, step, with_image: bool):
        obs = {k: v for k, v in env_obs.items() if k not in ('text', 'image')}
        obs['image'] = env_obs.get('image') if with_image else None
        obs['text'] = [template.format(env_prompt=t) for t in env_obs['text']]
        return obs

    def build_prompt(self, env_obs, team_context, step):
        return self._build_obs(self.prompt, env_obs, team_context, step, with_image=self.SEES_IMAGE)

    # -- sealed bid (the NEW skill; every agent, every step) ------------------------------------
    def bid_call(self, gen_batch: DataProto, env_obs: Dict[str, Any], team_context: List[str],
                 actor_rollout_wg, agent_active_mask, step: int) -> Tuple[DataProto, np.ndarray]:
        obs = self._build_obs(self._bid_prompt, env_obs, team_context, step,
                              with_image=(self.SEES_IMAGE and self._bid_image))
        batch = preprocess_batch(gen_batch=gen_batch, obs=obs, config=self.config,
                                 tokenizer=self.tokenizer, processor=self.processor)
        meta = {**gen_batch.meta_info,
                'sampling_overrides': {'max_new_tokens': self._bid_max_tokens, 'stop': ['</bid>']}}
        batch, text_responses = self._generate_with_llm(batch, actor_rollout_wg, agent_active_mask,
                                                        meta, kind="bid")
        bids, has_bid, _ = parse_open_bid(text_responses, self._bid_max)
        n = len(text_responses)
        batch.non_tensor_batch['bid'] = bids.astype(np.float32)
        batch.non_tensor_batch['raw_response'] = np.array(list(text_responses), dtype=object)
        batch.non_tensor_batch['is_action_valid'] = has_bid
        batch.non_tensor_batch['env_step'] = np.array([step] * n, dtype=object)
        batch.non_tensor_batch['call_kind'] = np.array(['bid'] * n, dtype=object)
        return batch, bids.astype(np.float32)

    # -- the winner's action (whitelist enforced by parser) -----------------------------------
    def call(self, gen_batch: DataProto, env_obs: Dict[str, Any], team_context: List[str],
             actor_rollout_wg, agent_active_mask, step: int) -> Tuple[DataProto, List[str]]:
        obs = self.build_prompt(env_obs, team_context, step)
        batch = preprocess_batch(gen_batch=gen_batch, obs=obs, config=self.config,
                                 tokenizer=self.tokenizer, processor=self.processor)
        batch, text_responses = self._generate_with_llm(batch, actor_rollout_wg, agent_active_mask,
                                                        gen_batch.meta_info, kind="act")
        emitted, instrs, verbs, oks = self.enforce(text_responses, agent_active_mask)
        n = len(text_responses)
        batch.non_tensor_batch['raw_response'] = np.array(list(text_responses), dtype=object)
        batch.non_tensor_batch['instruction'] = np.array(instrs, dtype=object)
        batch.non_tensor_batch['first_word'] = np.array(verbs, dtype=object)
        batch.non_tensor_batch['whitelist_ok'] = np.array(oks, dtype=bool)
        batch.non_tensor_batch['is_action_valid'] = np.array(oks, dtype=bool)
        batch.non_tensor_batch['env_step'] = np.array([step] * n, dtype=object)
        batch.non_tensor_batch['call_kind'] = np.array(['act'] * n, dtype=object)
        return batch, emitted

    # -- parser: enforce the agent's action slice (pure; GPU-free) -----------------------------
    def enforce(self, text_responses: List[str], mask=None):
        """Whitelist gate. The emitted text must name the agent's own action verb. A rejected row's
        emitted text carries a non-action string, so the env parser (is_valid_action) fails it and
        the step is wasted -- the agent already paid its winning bid, so the violation is priced."""
        n = len(text_responses)
        if mask is None:
            mask = np.ones(n, dtype=bool)
        emitted, instrs, verbs, oks = [], [], [], []
        for i in range(n):
            if not mask[i]:
                emitted.append(""); instrs.append(None); verbs.append(""); oks.append(False)
                continue
            txt = text_responses[i] if isinstance(text_responses[i], str) else ""
            # parse the ACTION inside <act>...</act>; the whole response (with <think>/<act>) is what
            # the env sees, and env-server parse_action reads the <act> span too.
            m = _ACT_TAG.search(txt)
            span = m.group(1) if m else ""
            ok = bool(m) and bool(self.SLICE_RE.search(span))
            kind = ("turn" if _TURN_RE.search(span) else "move forward" if _MOVE_RE.search(span)
                    else "stop" if _STOP_RE.search(span) else "")
            instrs.append(span.strip())
            verbs.append(kind)
            oks.append(ok)
            if ok:
                emitted.append(txt.strip())          # pass the full <think>+<act> response to env
            else:
                # The rejection string goes STRAIGHT to the habitat server: with
                # agent.multi_agent=True env_manager skips the projection/validity filter, and the
                # server's parse_action falls back to scanning the whole string when it finds no
                # <act> tag. A human-readable message naming this agent's own verbs therefore gets
                # EXECUTED as a real action (a Stopper's reject would end the episode). Emit a
                # tagged non-verb instead: parse_action narrows to the <act> span, finds no verb,
                # and takes its malformed branch -- no move, reward 0.0, one atomic step burned,
                # which is the intended "the turn is wasted and the agent still paid" pricing.
                # The readable reason survives in instruction/first_word/whitelist_ok for traces.
                emitted.append("<act>INVALID</act>")
        return emitted, instrs, verbs, oks


class _TurnerDesk(_NavilaDeskBase):
    DESK = "TURNER"
    VERBS = "turn left or turn right by 15, 30, or 45 degree"
    SLICE_RE = _TURN_RE
    HINT = ("You orient the robot. Use a turn to face the direction the instruction wants before the "
            "Mover advances. You cannot move forward or stop.")


class _MoverDesk(_NavilaDeskBase):
    DESK = "MOVER"
    VERBS = "move forward by 25, 50, or 75 cm"
    SLICE_RE = _MOVE_RE
    HINT = ("You make forward progress once the robot faces the right way. You cannot turn or stop.")


class _StopperDesk(_NavilaDeskBase):
    DESK = "STOPPER"
    VERBS = "stop"
    SLICE_RE = _STOP_RE
    HINT = ("You end the episode by stopping. Win ONLY when the robot has reached the goal the "
            "instruction describes -- stopping early or late both fail. You cannot turn or move.")


@AgentRegistry.register("Turner Desk (navila)")
class TurnerDeskAgent(_TurnerDesk):
    AGENT_NAME = "Turner Desk (navila)"


@AgentRegistry.register("Mover Desk (navila)")
class MoverDeskAgent(_MoverDesk):
    AGENT_NAME = "Mover Desk (navila)"


@AgentRegistry.register("Stopper Desk (navila)")
class StopperDeskAgent(_StopperDesk):
    AGENT_NAME = "Stopper Desk (navila)"


DESK_SLICES = {
    "Turner Desk (navila)": "turn",
    "Mover Desk (navila)": "move forward",
    "Stopper Desk (navila)": "stop",
}

__all__ = ["TurnerDeskAgent", "MoverDeskAgent", "StopperDeskAgent", "DESK_SLICES"]
