from __future__ import annotations

from agent_system.agent.orchestra.chain_base import ChainOrchestraBase


class OpenResearcherChainOrchestra(ChainOrchestraBase):
    """MPT auction over deep-research turns; one auction per run() call."""

    DOMAIN_BLOCK = "openresearcher_chain"
    AGENT_MODULE = "agent_system.agent.agents.openresearcher"
    ALT_COUNTER_ENV = "OR_ALT_COUNTER"
    TRACE_ENV = "OR_CHAIN_TRACE"
    LOG_TAG = "or-chain"
    SEG_SCALE_DEFAULT = False      # terminal-only reward: never scale by segment length

    def _after_turn(self, t, step, bs, active, res, bid_texts, acts, ctx):
        verb_of, ok_of, raw_of = acts["verb"], acts["ok"], acts["raw"]
        # trace (bids + winner + tool name + whitelist_ok + answered)
        if self._trace_path:
            recs = []
            for s in range(bs):
                if not active[s]:
                    continue
                w = int(res.winner[s])
                recs.append({
                    "kind": "turn", "pid": self._pid, "episode": self._ep_seq, "sample": int(s),
                    "train": self._is_train, "env_step": int(step), "turn": int(t),
                    "bids": {self.desk_ids[i]: round(float(res.raw_bids[i][s]), 4) for i in range(self.n)},
                    "winner": self.desk_ids[w], "price_paid": round(float(res.price[s]), 4),
                    "tool": verb_of[s], "whitelist_ok": bool(ok_of[s]), "answered": verb_of[s] == "answer", "action": raw_of[s],
                    "bid_texts": {self.desk_ids[i]: (str(bid_texts[i][s])[:self._trace_text]
                                                     if bid_texts.get(i) is not None else "")
                                  for i in range(self.n)},
                })
            self._trace(recs)

        nwin = int(active.sum())
        by_desk = {self.desk_ids[i]: int((res.chosen == i).sum()) for i in range(self.n)}
        passes = {self.desk_ids[i]: int((active & res.bidders[i] & (res.raw_bids[i] == 0.0)).sum())
                  for i in range(self.n)}
        print(f"[or-chain] turn={t} active={nwin} wins={by_desk} "
              f"whitelist_ok={int(sum(ok_of))}/{nwin} "
              + (f"passes={passes} " if passes else "")
              + f"bid_fallbacks={self._bid_fallbacks}", flush=True)

    def _after_finalize(self, bundle, bs, episode_rewards, is_train):
        self._episode_summary_trace(bs, episode_rewards)
