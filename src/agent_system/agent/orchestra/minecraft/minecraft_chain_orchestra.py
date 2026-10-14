from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Tuple

import numpy as np

from agent_system.agent.orchestra.chain_base import ChainOrchestraBase
from economy_engine.engine import settle_revenue_share as _engine_settle_rs

# ---- reading the world state back out of the env's text observation ---------------------------
# The env server (server/mc_env/mc_env_server.py::Slot._text) renders, per turn:
#     TASK: ...
#     Decision 7/120 · reward so far: 6
#     Totals so far: oak_log x5, oak_planks x4        <- cumulative mine_block/craft_item counters
#     What you have already done (...)                <- rolling history, MC_HISTORY turns
#     Inventory: oak_log x3, stick x4                 <- live inventory, MC_SCRIPTED_CRAFT=1 only
#     Last: executed 'chop the tree'; reward this step = 1
# `env_obs['anchor']` carries that text verbatim (env_manager.MinecraftEnvironmentManager._pack
# sets anchor=list(obs) BEFORE the prompt template is applied), so parsing it needs no extra
# plumbing and no change to the env package. The per-step `info` dict (which also carries
# 'delta'/'inventory') is NOT visible from inside run() -- it is returned by envs.step(), one
# layer above.
_INV_RE = re.compile(r"^Inventory:\s*(.*)$", re.M)
_TOT_RE = re.compile(r"^Totals so far:\s*(.*)$", re.M)
_ITEM_RE = re.compile(r"([A-Za-z_][A-Za-z_0-9]*)\s*x\s*(\d+)")
_CUM_RE = re.compile(r"reward so far:\s*([-+0-9.eE]+)")
_DEC_RE = re.compile(r"Decision\s+(\d+)\s*/\s*(\d+)")

# stone-pickaxe tech tree, in order. `milestone` = the deepest stage present in the state.
_MILESTONES: Tuple[Tuple[str, str], ...] = (
    ("log", "suffix:_log"),
    ("planks", "suffix:_planks"),
    ("crafting_table", "exact:crafting_table"),
    ("stick", "exact:stick"),
    ("wooden_pickaxe", "exact:wooden_pickaxe"),
    ("cobblestone", "exact:cobblestone"),
    ("stone_pickaxe", "exact:stone_pickaxe"),
)


def _parse_items(line: str) -> Dict[str, int]:
    return {m.group(1): int(m.group(2)) for m in _ITEM_RE.finditer(line or "")}


def parse_world_state(text: str) -> Dict[str, Any]:
    """{'inv': {...}, 'totals': {...}, 'state': {...}, 'cum_reward': float, 'decision': int}.

    'state' is the inventory when the server exposes one (MC_SCRIPTED_CRAFT=1) and the cumulative
    event totals otherwise, i.e. the best available view of what the player HAS."""
    text = text if isinstance(text, str) else ""
    m_inv, m_tot = _INV_RE.search(text), _TOT_RE.search(text)
    inv = _parse_items(m_inv.group(1)) if m_inv else {}
    tot = _parse_items(m_tot.group(1)) if m_tot else {}
    m_cum, m_dec = _CUM_RE.search(text), _DEC_RE.search(text)
    return {
        "inv": inv,
        "totals": tot,
        "state": inv if inv else tot,
        "cum_reward": float(m_cum.group(1)) if m_cum else None,
        "decision": int(m_dec.group(1)) if m_dec else None,
    }


def milestone_of(state: Dict[str, int]) -> Tuple[int, str]:
    """Deepest stone-pickaxe tech-tree stage present in `state`; (-1, 'none') when empty."""
    best, name = -1, "none"
    keys = list(state or {})
    for idx, (label, rule) in enumerate(_MILESTONES):
        kind, pat = rule.split(":", 1)
        hit = any(k.endswith(pat) for k in keys) if kind == "suffix" else (pat in keys)
        if hit:
            best, name = idx, label
    return best, name


class MinecraftChainOrchestra(ChainOrchestraBase):
    """MPT auction over Minecraft env turns; one auction per run() call. The training logic is
    ChainOrchestraBase; this class adds the world-state parsing and the Minecraft trace."""

    DOMAIN_BLOCK = "minecraft_chain"
    AGENT_MODULE = "agent_system.agent.agents.minecraft"
    ALT_COUNTER_ENV = "MC_ALT_COUNTER"
    TRACE_ENV = "MC_CHAIN_TRACE"
    LOG_TAG = "mc-chain"
    SEG_SCALE_DEFAULT = True

    def _init_domain(self):
        # Outcome records (step_reward / beta / profit per turn) are appended at finalize, aligned
        # with the decision records by (pid, episode, sample, turn). Append-only, never touches
        # control flow; off by default.
        self._trace_outcome = os.environ.get("MC_CHAIN_TRACE_OUTCOME", "") == "1"
        self._trace_max = int(os.environ.get("MC_CHAIN_TRACE_MAX_SAMPLES", "1000000"))
        # bid replies carry <think>; a short clip would cut them mid-sentence, which is useless
        # for showing why an agent bid what it bid.
        self._trace_text_max = int(os.environ.get("MC_CHAIN_TRACE_TEXT_MAX", "800"))

    def _turn_context(self, env_obs, bs):
        # world state BEFORE this turn's action == the state the agents are bidding on.
        raw = env_obs.get('anchor')
        if raw is None or len(raw) != bs:
            raw = env_obs['text']
        return [parse_world_state(raw[i]) for i in range(bs)]

    def _after_turn(self, t, step, bs, active, res, bid_texts, acts, ctx):
        # ---- trace + logging: env layer, driven purely by the numeric TurnResult ----
        ok_of = acts["ok"]
        recs0 = self._build_trace(t, step, bs, active, res.raw_bids, res.bid_vals, bid_texts,
                                  res.winner, res.price, acts["instr"], acts["verb"], ok_of,
                                  acts["raw"], ctx)
        _forced = getattr(res, "bid_forced", None)
        for r in recs0:
            # floor-reopened all-pass turn: the traced bids are synthetic (see auction.py)
            r["bid_forced"] = bool(_forced[r["sample"]]) if _forced is not None else False
        if recs0:
            self._trace(recs0)
        nwin = int(active.sum())
        by_desk = {self.desk_ids[i]: int((res.chosen == i).sum()) for i in range(self.n)}
        passes = {self.desk_ids[i]: int((active & res.bidders[i] & (res.raw_bids[i] == 0.0)).sum())
                  for i in range(self.n)}
        print(f"[mc-chain/rs] turn={t} active={nwin} wins={by_desk} "
              f"passes={passes} bid_fallbacks={self._bid_fallbacks} "
              f"beta[mean]={float(res.price[active].mean()) if nwin else 0.0:.2f} "
              f"whitelist_ok={int(sum(ok_of))}/{nwin}", flush=True)

    # ------------------------------------------------------------------------------------------
    def _build_trace(self, t, step, bs, active, raw_bids, bid_vals, bid_texts, winner, price_t,
                     instr_of, verb_of, ok_of, raw_of, state_now) -> List[dict]:
        """One record per (sample, turn) with the decision; the realised outcome is appended
        at finalize_episode when MC_CHAIN_TRACE_OUTCOME=1."""
        if not self._trace_path:
            return []
        prev_win = None
        if t >= 1:
            prev_win = self._chosen[t - 1]
        out = []
        for s in range(min(bs, self._trace_max)):
            if not active[s]:
                continue
            w = int(winner[s])
            pw = int(prev_win[s]) if prev_win is not None else -1
            st = state_now[s]
            mi, mname = milestone_of(st["state"])
            out.append({
                "kind": "turn", "pid": self._pid, "episode": self._ep_seq, "sample": int(s),
                "train": self._is_train, "env_step": int(step), "turn": int(t),
                "bids": {self.desk_ids[i]: round(float(raw_bids[i][s]), 4) for i in range(self.n)},
                "eff_bids": {self.desk_ids[i]: round(float(bid_vals[i][s]), 4)
                             for i in range(self.n) if np.isfinite(bid_vals[i][s])},
                "bid_texts": {self.desk_ids[i]: (str(bid_texts[i][s])[:self._trace_text_max]
                                                 if bid_texts.get(i) is not None else "")
                              for i in range(self.n)},
                "winner": self.desk_ids[w],
                "winning_bid": round(float(raw_bids[w][s]), 4),
                "price_paid": round(float(price_t[s]), 4),
                # turn 0 has no upstream agent to pass a share back to.
                "paid_to": (self.desk_ids[pw] if pw >= 0 else None),
                "instruction": instr_of[s], "first_word": verb_of[s],
                "whitelist_ok": bool(ok_of[s]), "raw_response": raw_of[s],
                "inv_before": st["state"], "totals_before": st["totals"],
                "cum_reward_before": st["cum_reward"], "decision": st["decision"],
                "milestone_before": mname, "milestone_before_idx": mi,
                # outcome placeholders (the per-turn outcome is written as separate "outcome"
                # records at finalize_episode when MC_CHAIN_TRACE_OUTCOME=1)
                "inv_after": None, "inv_delta": None, "milestone": None, "milestone_idx": None,
                "milestone_advanced": None, "cum_reward_after": None, "step_reward": None,
                "terminal": None, "episode_reward": None, "econ_credit": None,
            })
        return out

    # ------------------------------------------------------------------------------------------
    @staticmethod
    def settle_revenue_share(betas: List[float], rewards: List[float], seg_scale: bool = True):
        """Delegates to economy_engine.engine.settle_revenue_share."""
        return _engine_settle_rs(betas, rewards, seg_scale)

    def _trace_rs_outcomes(self, bundle, bs: int, episode_rewards, is_train: bool):
        """Append the outcome records for the revenue-share settlement the engine just ran.
        Append-only outcome records aligned with the decision records by (pid, episode,
        sample, turn); off unless MC_CHAIN_TRACE_OUTCOME=1."""
        if not self._trace_outcome:
            return
        try:                        # a trace write must never kill a rollout (same as _trace)
            T, profit, R_steps = bundle["T"], bundle["profit"], bundle["R_steps"]
            orecs = []
            for s_i in range(bs):
                last = int(self._term_at[s_i]) if self._term_at is not None else T - 1
                if last < 0:
                    continue
                er = None
                if episode_rewards is not None:
                    er = float(np.asarray(episode_rewards).reshape(-1)[s_i])
                for t in range(last + 1):
                    orecs.append({"kind": "outcome", "pid": self._pid, "episode": self._ep_seq,
                                  "sample": int(s_i), "turn": t, "train": bool(is_train),
                                  "step_reward": float(R_steps[t][s_i]),
                                  "beta": float(self._betas[t][s_i]),
                                  "profit": float(profit[t, s_i]),
                                  "terminal_turn": last, "episode_reward": er})
            self._trace(orecs)
        except Exception as e:
            print(f"[mc-chain] outcome trace failed (non-fatal): {e}", flush=True)

    def _after_finalize(self, bundle, bs, episode_rewards, is_train):
        self._trace_rs_outcomes(bundle, bs, episode_rewards, is_train)
