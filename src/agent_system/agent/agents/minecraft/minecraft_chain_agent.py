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
from agent_system.environments.env_package.minecraft.envs import extract_instruction

# <image> MUST be present whenever an image is attached: the VLM path attaches pixel_values, and a
# prompt without an image token crashes the forward pass on mixed-modality batches.
_IMAGE_HEADER = "# Screenshot\n<image>\n\n"

# The bid prompt (five levels, 0 = pass: the agent sits the auction out, cannot win, pays nothing).
# The format example shows ONE concrete number (a "<bid>0.2|0.4|0.6|0.8</bid>" example teaches
# weak-format agents to echo the menu verbatim), and the scale is anchored to what the agent's VERBS
# can do from the exact inventory in front of it. Byte-identical to the agent SFT corpora.
BID_BODY_V24 = (
    "{env_prompt}\n\n"
    "# You are the {desk} desk\n"
    "You are one of {n_desks} competing desks that share control of this single Minecraft player.\n"
    "Exactly ONE desk acts each turn, decided by a sealed-bid auction that you are bidding in now.\n"
    "If you win you must issue an instruction whose FIRST WORD is one of: {verbs}.\n"
    "You cannot issue any other kind of instruction, so a turn you win that your verbs cannot\n"
    "advance is a turn wasted for the whole team, and you still pay the rent.\n"
    "\n"
    "# Bidding (revenue share, five levels)\n"
    "Your bid is the share of your income this turn that you promise to pass back to the desk\n"
    "that acted before you (you are buying the state it produced). Your income is what the\n"
    "environment pays your action plus the share the NEXT winner passes back to you, so value\n"
    "flows backward along the whole chain. The HIGHEST bid wins the turn; only the winner pays\n"
    "a small rent.\n"
    "First read the Inventory and 'Craftable now' lines literally and decide whether YOUR verbs\n"
    "can advance the {chain} chain from THIS exact state. Then bid:\n"
    "  0    = pass. My verbs cannot help right now; another desk must act first.\n"
    "  0.2  = my verbs barely help here.\n"
    "  0.4  = my verbs are useful here.\n"
    "  0.6  = my verbs are important here.\n"
    "  0.8  = I am the bottleneck desk: the chain cannot move without my verbs right now.\n"
    "Your bid MUST be exactly one of: 0, 0.2, 0.4, 0.6, 0.8 (no other numbers).\n"
    "Answer with ONE short sentence of reasoning that cites concrete inventory facts, then the\n"
    "number, for example:\n"
    "<think>one sentence about what is in the inventory and what my verbs can do with it</think>\n"
    "<bid>0.4</bid>\n"
)

ACT_BODY = (
    "{env_prompt}\n\n"
    "# You are the {desk} desk and you WON this turn's auction\n"
    "You issue the single instruction for this turn. A low-level controller (STEVE-1) executes it\n"
    "for about two seconds of game time; the crafting verbs are executed exactly by the server.\n"
    "\n"
    "HARD CONSTRAINT, enforced by a parser and not negotiable: the FIRST WORD of your instruction\n"
    "must be one of: {verbs}\n"
    "Anything else is REJECTED before it reaches the game: the turn is wasted, nothing happens,\n"
    "and you still pay your promised share. {desk_hint}\n"
    "\n"
    "Keep it short, concrete and visual. Example: <action>{example}</action>\n"
    "Reply with your reasoning, then the instruction wrapped EXACTLY like this:\n"
    "<action>your short instruction here</action>\n"
)




def first_verb(instruction: str) -> str:
    """Lower-cased first alphabetic token of an instruction ('' when there is none)."""
    if not instruction:
        return ""
    m = re.search(r"[A-Za-z]+", instruction)
    return m.group(0).lower() if m else ""


class _MinecraftDeskBase(BaseAgent):
    AGENT_NAME = "Desk"
    DESK = "Desk"
    VERBS: Tuple[str, ...] = ()
    EXAMPLE = "look around"
    HINT = ""
    SEES_IMAGE = True

    def __init__(self, wg_id: str, tokenizer: PreTrainedTokenizer, processor, config: Any):
        cfg = getattr(getattr(getattr(config, "agent", None), "orchestra", None), "minecraft_chain", None)

        def _get(k, d):
            return getattr(cfg, k, d) if cfg is not None else d

        self._bid_max = float(_get("bid_max", 1.0))
        # Which crafting chain this society is working: names the goal in the bid prompt.
        self._chain = str(_get("chain_name", "iron-pickaxe"))
        # 16 tokens was enough for a bare "<bid>0.83</bid>", but the agents are also trained to
        # think before acting, and under the rollout's SAMPLING (the offline greedy check does not
        # show this) every probe opened with "<think>" and was truncated before the tag appeared.
        # Every bid then parsed to 0.0 and the auction degenerated into a random dispatcher -- the
        # market was never actually being exercised. The bid prompt now asks for one short sentence
        # first, so the budget has to cover it.
        self._bid_max_tokens = int(_get("bid_max_tokens", 96))
        # The spec is explicit that every agent sees the FULL observation, so the bid probe carries
        # the screenshot too (default). bid_sees_image=False is a pure COST lever (3 fewer vision
        # forwards per env turn) that changes the experiment -- it makes the bid a text-only
        # valuation -- so it is off by default and must be turned on deliberately.
        self._bid_image = bool(_get("bid_sees_image", True))
        try:
            self._n_desks = len(config.agent.agent_ids)
        except Exception:
            self._n_desks = 3
        # Substitute the agent-constant fields ONCE, here, with plain .replace() -- NOT .format(),
        # which would also try to expand {env_prompt} (and would choke on any brace a hint or an
        # env observation happens to contain). Only {env_prompt} survives into the per-turn
        # .format() in _build_obs.
        def _fill(body: str) -> str:
            for k, v in (("{desk}", self.DESK), ("{verbs}", " | ".join(self.VERBS)),
                         ("{bid_max}", f"{self._bid_max:g}"), ("{example}", self.EXAMPLE),
                         ("{desk_hint}", self.HINT), ("{n_desks}", str(self._n_desks)),
                         ("{chain}", self._chain)):
                body = body.replace(k, v)
            return body

        # MPT bids with bid_pass=True: the five-level prompt (0 = sit the auction out).
        self._bid_pass = bool(_get("bid_pass", True))
        assert self._bid_pass, "MPT uses bid_pass=True (level 0 = pass)"
        self._bid_prompt = (_IMAGE_HEADER if self._bid_image else "") + _fill(BID_BODY_V24)
        act_prompt = _IMAGE_HEADER + _fill(ACT_BODY)
        super().__init__(self.AGENT_NAME, act_prompt, wg_id=wg_id, tokenizer=tokenizer,
                         processor=processor, config=config)

    # -- helpers ----------------------------------------------------------------
    def _build_obs(self, template: str, env_obs, team_context, step, with_image: bool):
        """Shallow rebuild (never deepcopy: env_obs['image'] holds PIL frames and this runs up to
        4x per env turn). `with_image` must agree with whether `template` contains <image>."""
        obs = {k: v for k, v in env_obs.items() if k not in ('text', 'image')}
        obs['image'] = env_obs.get('image') if with_image else None
        obs['text'] = [template.format(env_prompt=t, team_context=team_context[i], step=step)
                       for i, t in enumerate(env_obs['text'])]
        return obs

    def build_prompt(self, env_obs, team_context, step):
        return self._build_obs(self.prompt, env_obs, team_context, step, with_image=self.SEES_IMAGE)

    # -- cheap length-capped bid probe (EVERY agent, every turn) -------------------
    def bid_call(self, gen_batch: DataProto, env_obs: Dict[str, Any], team_context: List[str],
                 actor_rollout_wg, agent_active_mask, step: int) -> Tuple[DataProto, np.ndarray]:
        obs = self._build_obs(self._bid_prompt, env_obs, team_context, step,
                              with_image=(self.SEES_IMAGE and self._bid_image))
        batch = preprocess_batch(gen_batch=gen_batch, obs=obs, config=self.config,
                                 tokenizer=self.tokenizer, processor=self.processor)
        meta = {**gen_batch.meta_info,
                'sampling_overrides': {'max_new_tokens': self._bid_max_tokens, 'stop': ['</bid>']}}
        batch, text_responses = self._generate_with_llm(batch, actor_rollout_wg, agent_active_mask, meta, kind="bid")
        bids, has_bid, _ = parse_open_bid(text_responses, self._bid_max)
        n = len(text_responses)
        batch.non_tensor_batch['bid'] = bids.astype(np.float32)
        batch.non_tensor_batch['raw_response'] = np.array(list(text_responses), dtype=object)
        # a probe that never quoted a price is a malformed action for the invalid-action penalty
        batch.non_tensor_batch['is_action_valid'] = has_bid
        batch.non_tensor_batch['env_step'] = np.array([step] * n, dtype=object)
        batch.non_tensor_batch['call_kind'] = np.array(['bid'] * n, dtype=object)
        return batch, bids.astype(np.float32)

    # -- the winner's instruction (whitelist enforced HERE, by parser) ------------
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
        # `is_action_valid` feeds actor.use_invalid_action_penalty: a whitelist violation IS an
        # invalid action (the env will reject it), so the two flags are the same thing here.
        batch.non_tensor_batch['is_action_valid'] = np.array(oks, dtype=bool)
        batch.non_tensor_batch['env_step'] = np.array([step] * n, dtype=object)
        batch.non_tensor_batch['call_kind'] = np.array(['act'] * n, dtype=object)
        return batch, emitted

    # -- the parser (pure; unit-tested without a GPU) -----------------------------
    def enforce(self, text_responses: List[str], mask=None):
        """Whitelist gate. Returns (emitted, instruction, first_word, ok) per row.

        A rejected row's emitted text carries NO <action>...</action>, so
        `minecraft/envs.extract_instruction` returns None and the env server answers
        "INVALID: you must emit <action>...</action>" with reward 0 and a consumed turn. The agent
        has already paid its winning bid, so the violation is priced, not merely logged.
        Accepted rows are re-wrapped canonically, so what the server parses is exactly the
        whitelisted instruction (no stray prose before the tag can shift the first word)."""
        n = len(text_responses)
        if mask is None:
            mask = np.ones(n, dtype=bool)
        emitted, instrs, verbs, oks = [], [], [], []
        for i in range(n):
            if not mask[i]:
                emitted.append(""); instrs.append(None); verbs.append(""); oks.append(False)
                continue
            instr = extract_instruction(text_responses[i] if isinstance(text_responses[i], str) else "")
            fw = first_verb(instr)
            ok = bool(instr) and fw in self.VERBS
            instrs.append(instr)
            verbs.append(fw)
            oks.append(ok)
            if ok:
                emitted.append(f"<action>{instr}</action>")
            else:
                emitted.append(f"[whitelist-reject] the {self.DESK} desk may only emit instructions "
                               f"whose first word is one of {'|'.join(self.VERBS)}; got {fw!r}")
        return emitted, instrs, verbs, oks


class _LoggerDesk(_MinecraftDeskBase):
    DESK = "LOGGER"
    # This exact tuple is what the agent SFT corpora were built against.
    VERBS = ("chop", "get", "gather", "break", "find", "scatter")
    EXAMPLE = "chop the tree in front of you"
    HINT = ("You are the surface-gathering desk: wood and anything reachable by walking up to it "
            "and breaking it. 'scatter' teleports the player a short distance -- use it only when "
            "the spawn has nothing usable in sight.")


class _MinerDesk(_MinecraftDeskBase):
    """The underground agent."""
    DESK = "MINER"
    VERBS = ("dig", "mine", "explore", "tunnel")
    EXAMPLE = "dig down to find stone"
    HINT = "You are the underground desk: getting below the surface and mining stone and ores."


class _SmithDesk(_MinecraftDeskBase):
    DESK = "SMITH"
    VERBS = ("craft", "equip", "smelt")
    EXAMPLE = "craft a stone pickaxe"
    HINT = ("You are the inventory desk: recipes and equipping. These verbs are executed exactly "
            "by the server, so a correctly-timed craft never fails for motor reasons -- but a "
            "craft with the ingredients missing fails outright and wastes the turn.")


@AgentRegistry.register("Logger Desk (chain)")
class LoggerDeskAgent(_LoggerDesk):
    AGENT_NAME = "Logger Desk (chain)"


@AgentRegistry.register("Miner Desk (chain)")
class MinerDeskAgent(_MinerDesk):
    AGENT_NAME = "Miner Desk (chain)"


@AgentRegistry.register("Smith Desk (chain)")
class SmithDeskAgent(_SmithDesk):
    AGENT_NAME = "Smith Desk (chain)"


DESK_VERBS = {
    "Logger Desk (chain)": _LoggerDesk.VERBS,
    "Miner Desk (chain)": _MinerDesk.VERBS,
    "Smith Desk (chain)": _SmithDesk.VERBS,
}

__all__ = ["LoggerDeskAgent", "MinerDeskAgent", "SmithDeskAgent", "DESK_VERBS", "first_verb"]
