from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from transformers import PreTrainedTokenizer

from agent_system.agent.registry import AgentRegistry
from agent_system.agent.utils import parse_open_bid
from agent_system.environments.env_package.openresearcher.action import encode_action
from agent_system.environments.env_package.openresearcher.parser import cut_at_stop, normalize_tool_name, parse_assistant_turn
from agent_system.multi_turn_rollout.utils import preprocess_batch
from agent_system.agent.agents.openresearcher.response_mask import trim_response_mask
from verl import DataProto

from .or_desk_base import _cfg_get, _OrAgentBase

ACT_SUFFIX = (
    "You are the {desk} desk, one of {n_desks} specialists ({desks}) that share this browser and "
    "this conversation. Exactly one desk acts each turn, chosen by an auction, and this turn is "
    "yours. You may ONLY call {tools}, or give the final answer with no tool call. Any other tool "
    "call is rejected and the turn is wasted. {hint}"
)

# The act prompt's trailing USER turn. Without it act and bid prompts differ only by the bid
# probe, and a Searcher often answered its first turn like a bid: one <think> and a stop, which the
# env takes as an empty final answer (RL traces: 6 of 25 first turns).
ACT_BODY = (
    "# Your turn: you are the {desk} desk\n"
    "You won this turn's auction, so you act now. Reason briefly inside <think></think>, then do "
    "exactly ONE of:\n"
    "  (a) call {tools}, written as\n"
    "<tool_call>\n{\"name\": \"{example_tool}\", \"arguments\": {...}}\n</tool_call>\n"
    "  (b) if the evidence in the conversation already answers the question, write the final "
    "answer with no tool call.\n"
    "A reply with only <think> and neither a tool call nor an answer counts as an empty final "
    "answer and scores 0. Calling any tool other than {tools} is rejected and wastes the turn.\n"
)

BID_BODY = (
    "# You are the {desk} desk\n"
    "You are one of {n_desks} competing desks ({desks}) that share control of this research "
    "session. Exactly ONE desk acts each turn, decided by a sealed-bid auction that you are bidding "
    "in now. If you win you may only call {tools}, or give the final answer. A turn you win that "
    "your tools cannot advance is a turn wasted for the whole team, and you still pay the rent.\n"
    "\n"
    "# Bidding (revenue share, five levels)\n"
    "Your bid is the share of your income this turn that you promise to pass back to the desk "
    "that acted before you (you are buying the state it produced). Your income is what the final "
    "answer earns plus the share the NEXT winner passes back to you, so value flows backward along "
    "the whole chain. The HIGHEST bid wins the turn; only the winner pays a small rent.\n"
    "Read the conversation so far literally -- what has been searched, which pages were opened, "
    "what evidence is already on screen -- and decide whether YOUR tools can advance the answer "
    "from THIS exact state. Then bid:\n"
    "  0    = pass. My tools cannot help right now; another desk must act first.\n"
    "  0.2  = my tools barely help here.\n"
    "  0.4  = my tools are useful here.\n"
    "  0.6  = my tools are important here.\n"
    "  0.8  = I am the bottleneck desk: the answer cannot move without my tools right now.\n"
    "Your bid MUST be exactly one of: 0, 0.2, 0.4, 0.6, 0.8 (no other numbers).\n"
    "Answer with ONE short sentence of reasoning that cites what is already in the conversation, "
    "then the number, for example:\n"
    "<think>one sentence about the evidence so far and what my tools can do next</think>\n"
    "<bid>0.4</bid>\n"
)


def desk_roster(agent_ids) -> str:
    """'Searcher Desk (openresearcher)', 'Opener Desk (openresearcher)' -> 'Searcher/Opener'.
    The {desks} fill of the prompt bodies: derived from the launcher's agent list, never hardcoded."""
    names = []
    for a in agent_ids or ():
        n = str(a).split(" Desk", 1)[0].strip()
        names.append(n or str(a))
    return "/".join(names)


class _OrDeskBase(_OrAgentBase):
    DESK = "DESK"
    TOOLS: Tuple[str, ...] = ()
    HINT = ""
    DEFAULT_DESKS = ("Searcher Desk (openresearcher)", "Opener Desk (openresearcher)", "Finder Desk (openresearcher)")

    def __init__(self, wg_id: str, tokenizer: PreTrainedTokenizer, processor, config: Any):
        cfg = getattr(getattr(getattr(config, "agent", None), "orchestra", None), "openresearcher_chain", None)
        self._bid_max = float(_cfg_get(cfg, "bid_max", 1.0))
        self._bid_max_tokens = int(_cfg_get(cfg, "bid_max_tokens", 64))
        try:
            agent_ids = list(config.agent.agent_ids)
        except Exception:
            agent_ids = list(self.DEFAULT_DESKS)
        self._n_desks = len(agent_ids)
        self.ALL_DESKS = desk_roster(agent_ids)
        fills = (("{desk}", self.DESK), ("{n_desks}", str(self._n_desks)), ("{desks}", self.ALL_DESKS),
                 ("{tools}", " or ".join(self.TOOLS)), ("{hint}", self.HINT),
                 ("{example_tool}", self.TOOLS[0] if self.TOOLS else ""))

        def _fill(body: str) -> str:
            for k, v in fills:
                body = body.replace(k, v)
            return body.strip()

        self.SYSTEM_SUFFIX = _fill(ACT_SUFFIX)
        self.ACT_TRAILING = _fill(ACT_BODY)
        self._bid_body = _fill(BID_BODY)
        super().__init__(wg_id=wg_id, tokenizer=tokenizer, processor=processor, config=config)

    def _tools_of(self, env_obs):
        """Only this agent's slice of the env's tool schemas is rendered into its prompt (act AND
        bid): listing all three made off-slice calls look legal (in RL, ~20 % of won turns
        called another agent's tool). The whitelist gate in enforce() stays the authority."""
        tools = env_obs.get("tools")
        if not tools:
            return tools
        return [t for t in tools if ("browser." + normalize_tool_name(str((t.get("function") or t).get("name", ""))).lower()) in self.TOOLS]

    # -- sealed bid (the NEW skill; every agent, every turn) ------------------------------------
    def bid_call(self, gen_batch: DataProto, env_obs: Dict[str, Any], team_context: List[str],
                 actor_rollout_wg, agent_active_mask, step: int) -> Tuple[DataProto, np.ndarray]:
        obs = self._build_obs(env_obs, trailing_user=self._bid_body)
        batch = preprocess_batch(gen_batch=gen_batch, obs=obs, config=self.config,
                                 tokenizer=self.tokenizer, processor=self.processor)
        meta = {**gen_batch.meta_info,
                'sampling_overrides': {'max_new_tokens': self._bid_max_tokens, 'stop': ['</bid>']}}
        batch, text_responses = self._generate_with_llm(batch, actor_rollout_wg, agent_active_mask,
                                                        meta, kind="bid")
        trim_response_mask(batch, self.tokenizer.pad_token_id, f"{self.name}/bid")   # `</bid>` stop: no EOS
        text_responses, _ = self._texts_and_caps(batch, text_responses)
        bids, has_bid, _ = parse_open_bid(text_responses, self._bid_max)
        n = len(text_responses)
        batch.non_tensor_batch['bid'] = bids.astype(np.float32)
        batch.non_tensor_batch['raw_response'] = np.array(list(text_responses), dtype=object)
        batch.non_tensor_batch['is_action_valid'] = has_bid
        batch.non_tensor_batch['env_step'] = np.array([step] * n, dtype=object)
        batch.non_tensor_batch['call_kind'] = np.array(['bid'] * n, dtype=object)
        return batch, bids.astype(np.float32)

    # -- the winner's turn (whitelist enforced by parser; env produces the feedback) -----------
    def call(self, gen_batch: DataProto, env_obs: Dict[str, Any], team_context: List[str],
             actor_rollout_wg, agent_active_mask, step: int) -> Tuple[DataProto, List[str]]:
        obs = self.build_prompt(env_obs, team_context, step)
        batch = preprocess_batch(gen_batch=gen_batch, obs=obs, config=self.config,
                                 tokenizer=self.tokenizer, processor=self.processor)
        batch, text_responses = self._generate_with_llm(batch, actor_rollout_wg, agent_active_mask,
                                                        {**gen_batch.meta_info, 'sampling_overrides': {'stop': ['<tool_response>']}},
                                                        kind="act")
        trim_response_mask(batch, self.tokenizer.pad_token_id, f"{self.name}/act")
        self._report_native_markup(batch, agent_active_mask, "act")
        text_responses, caps = self._texts_and_caps(batch, text_responses)
        emitted, instrs, verbs, oks = self.enforce(text_responses, agent_active_mask, caps=caps)
        n = len(text_responses)
        batch.non_tensor_batch['raw_response'] = np.array(list(text_responses), dtype=object)
        batch.non_tensor_batch['instruction'] = np.array(instrs, dtype=object)
        batch.non_tensor_batch['first_word'] = np.array(verbs, dtype=object)
        batch.non_tensor_batch['whitelist_ok'] = np.array(oks, dtype=bool)
        # a whitelist violation IS an invalid action (the env rejects it) -> invalid-action penalty
        batch.non_tensor_batch['is_action_valid'] = np.array(oks, dtype=bool)
        batch.non_tensor_batch['env_step'] = np.array([step] * n, dtype=object)
        batch.non_tensor_batch['call_kind'] = np.array(['act'] * n, dtype=object)
        return batch, emitted

    # -- parser: enforce the agent's tool slice (pure; GPU-free) --------------------------------
    def enforce(self, text_responses: List[str], mask=None, caps=None):
        """Whitelist gate. Returns (emitted, instruction, first_word, ok) per row.

        The emitted text is ALWAYS the raw turn (stop string cut), wrapped in the action envelope
        that carries this agent's whitelist: the env appends the raw turn to history and answers an
        off-slice tool call with OpenResearcher's own "Tool X not available" message, so the
        turn is burned and the agent still paid its winning bid. `first_word` is the tool's full
        name, "answer" for a final answer, or "parse_error"."""
        n = len(text_responses)
        if mask is None:
            mask = np.ones(n, dtype=bool)
        emitted, instrs, verbs, oks = [], [], [], []
        allowed = sorted(self.TOOLS)
        for i in range(n):
            if not mask[i]:
                emitted.append(""); instrs.append(None); verbs.append(""); oks.append(False)
                continue
            txt = cut_at_stop(text_responses[i] if isinstance(text_responses[i], str) else "")
            p = parse_assistant_turn(txt)
            if not p.has_tool_call:
                ok, verb = True, "answer"
            elif p.parse_error:
                ok, verb = False, "parse_error"
            else:
                # canonical full name: the env (execute_tool) accepts OR's short spelling
                # ("search" == "browser.search"), so the whitelist verdict must too, or a
                # short-name call is executed by the env yet stamped invalid here
                verb = "browser." + normalize_tool_name(str(p.name or "")).lower()
                ok = verb in self.TOOLS
            instrs.append(p.tool_call_text)
            verbs.append(verb)
            oks.append(ok)
            emitted.append(encode_action(txt, allowed=allowed, hit_cap=(caps[i] if caps is not None else None)))
        return emitted, instrs, verbs, oks


class _SearcherDesk(_OrDeskBase):
    DESK = "SEARCHER"
    TOOLS = ("browser.search",)
    HINT = ("You decide what to look for: craft queries that decompose the question into "
            "findable facts. You cannot open or read pages.")


class _OpenerDesk(_OrDeskBase):
    DESK = "OPENER"
    TOOLS = ("browser.open",)
    HINT = ("You open pages: follow link ids from the search results or open URLs directly, and "
            "scroll with loc / num_lines to read further. You cannot run searches or find text in a page.")


class _FinderDesk(_OrDeskBase):
    DESK = "FINDER"
    TOOLS = ("browser.find",)
    HINT = ("You search inside the page that is currently open: find a pattern to confirm or rule "
            "out a candidate fact. You cannot run web searches or open pages.")


@AgentRegistry.register("Searcher Desk (openresearcher)")
class SearcherDeskAgent(_SearcherDesk):
    AGENT_NAME = "Searcher Desk (openresearcher)"


@AgentRegistry.register("Opener Desk (openresearcher)")
class OpenerDeskAgent(_OpenerDesk):
    AGENT_NAME = "Opener Desk (openresearcher)"


@AgentRegistry.register("Finder Desk (openresearcher)")
class FinderDeskAgent(_FinderDesk):
    AGENT_NAME = "Finder Desk (openresearcher)"


DESK_TOOLS = {
    "Searcher Desk (openresearcher)": _SearcherDesk.TOOLS,
    "Opener Desk (openresearcher)": _OpenerDesk.TOOLS,
    "Finder Desk (openresearcher)": _FinderDesk.TOOLS,
}
DEFAULT_DESKS = _OrDeskBase.DEFAULT_DESKS       # the 3-agent roster the launchers use

__all__ = ["SearcherDeskAgent", "OpenerDeskAgent", "FinderDeskAgent",
           "DESK_TOOLS", "DEFAULT_DESKS", "desk_roster"]
