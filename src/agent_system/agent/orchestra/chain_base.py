from __future__ import annotations

import importlib
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from transformers import PreTrainedTokenizer

from verl import DataProto
from agent_system.agent.orchestra.base import BaseOrchestra
from economy_engine.engine import AlternatingPhase, ChainTurnEngine


def resolve_alt_counter(domain_env: str) -> str:
    """Path of the alternation counter file: MPT_ALT_COUNTER, else the domain's own variable
    (MC_ALT_COUNTER / NAVILA_ALT_COUNTER / OR_ALT_COUNTER, kept for existing launchers)."""
    return os.environ.get("MPT_ALT_COUNTER") or os.environ.get(domain_env, "")


def checkpoint_iteration(trainer_cfg) -> Optional[int]:
    """The step the trainer resumes from under trainer.resume_mode=auto: the integer in
    {trainer.default_local_dir}/latest_checkpointed_iteration.txt, or None (no checkpoint, another
    resume mode, unreadable file)."""
    try:
        if str(trainer_cfg.get("resume_mode", "auto")) != "auto":
            return None
        d = trainer_cfg.get("default_local_dir", None)
        if not d:
            return None
        path = os.path.join(str(d), "latest_checkpointed_iteration.txt")
        if not os.path.isfile(path):
            return None
        digits = re.sub(r"[^0-9]", "", open(path).read())
        return int(digits) if digits else None
    except Exception:                                             # noqa: BLE001
        return None


class ChainOrchestraBase(BaseOrchestra):
    """MPT auction over env turns; one auction per run() call. Subclasses set the class
    attributes below and may override the _init_domain / _turn_context / _after_turn /
    _after_finalize hooks."""

    DOMAIN_BLOCK = ""             # agent.orchestra.<DOMAIN_BLOCK> config block
    AGENT_MODULE = ""             # module that registers the domain's agents
    ALT_COUNTER_ENV = ""          # per-domain fallback name of the alternation counter variable
    TRACE_ENV = ""                # per-domain decision-trace path variable (<TRACE_ENV>_TEXT: clip)
    LOG_TAG = "chain"             # prefix of the log lines
    SEG_SCALE_DEFAULT = True      # reward-shape default of the settlement (see the yaml block)

    def __init__(self, agent_ids: List[str], model_ids: List[str], agents_to_wg_mapping: Dict[str, str],
                 tokenizers: Dict[str, PreTrainedTokenizer] = None, processors: Dict[str, Any] = None,
                 config: Any = None):
        importlib.import_module(self.AGENT_MODULE)
        super().__init__(agent_ids=agent_ids, model_ids=model_ids,
                         agents_to_wg_mapping=agents_to_wg_mapping, tokenizers=tokenizers,
                         processors=processors, config=config)
        if len(self.agents) < 2:
            raise ValueError(f"{type(self).__name__} needs >=2 competing agents; "
                             f"got agent_ids={agent_ids}.")
        self.desk_ids: List[str] = list(self.agent_ids)
        self.n = len(self.desk_ids)

        cfg = getattr(getattr(self.config.agent, "orchestra", None), self.DOMAIN_BLOCK, None)
        self._cfg = cfg

        def _get(k, d):
            return getattr(cfg, k, d) if cfg is not None else d

        self.bid_max = float(_get("bid_max", 1.0))
        # Counterfactual preference mode (cf_preference=True): bid-phase iterations of the act/bid
        # alternation train an answer-level preference loss on counterfactual branch pairs
        # instead of policy gradient (see _alt_apply and counterfactual_collector). The
        # collector object lives here so rollout_loop can reach it via the orchestra; it is
        # inert unless cf_active is raised for the iteration.
        self.cf_preference = bool(_get("cf_preference", False))
        self.cf_active = False
        self.cf_turn_candidates = None
        self.cf_collector = None
        if self.cf_preference:
            # cf_active fires on BID-phase iterations, so the alternation must actually be
            # on. In "fixed" mode (both iters 0) train_bid_rows is True on EVERY iteration
            # -- cf_active would then be permanently raised and the run would never take a
            # single policy-gradient step. Refuse loudly instead of training nothing.
            if int(_get("alt_act_iters", 0)) <= 0 or int(_get("alt_bid_iters", 0)) <= 0:
                raise ValueError(
                    f"{self.DOMAIN_BLOCK}.cf_preference=True requires the act/bid alternation: set "
                    "alt_act_iters>0 and alt_bid_iters>0 (e.g. 4 and 1). In fixed mode every "
                    "iteration would be a preference iteration and policy gradient would never "
                    "run.")
        # R is divided by this before it enters the settlement.
        self.reward_scale = float(_get("reward_scale", 1.0))
        if self.reward_scale <= 0:
            raise ValueError(f"{self.DOMAIN_BLOCK}.reward_scale must be > 0")
        # ONE participation rent, charged only to the agent that WINS the turn (on its bid row
        # and on its act row, both settled at profit - rent). MPT requires rent > 0.
        self.reward_rent = float(_get("reward_rent", 0.05))
        # Revenue-share settlement is hard-wired in MPT (see the module docstring).
        self.revenue_share = bool(_get("revenue_share", True))
        assert self.revenue_share, "MPT requires revenue_share=True"
        # Whether bid probes are training rows (the configured default; the alternation
        # overwrites it every training iteration).
        self.train_bid_rows = bool(_get("train_bid_rows", True))
        # ---- act/bid alternation (how-to-act / how-to-bid) --------------------------------
        # Several iterations train only the act rows, then one iteration trains only the bid
        # rows, cyclically, so the two heads never share one gradient step. train_bid_rows /
        # train_act_rows are re-derived every training iteration from the position in the
        # (act_iters, bid_iters) cycle; both 0 = fixed mode (the configured values). The
        # iteration index comes from a counter file outside the process (MPT_ALT_COUNTER, or the
        # domain's own variable): the orchestra cannot see the trainer's global_step. The phase
        # is resolved in _alt_apply(), not here.
        self.alt_act_iters = int(_get("alt_act_iters", 0))
        self.alt_bid_iters = int(_get("alt_bid_iters", 0))
        self._alt_counter = resolve_alt_counter(self.ALT_COUNTER_ENV)
        self.alt_phase = "fixed"
        self._alt_bid_cfg = self.train_bid_rows   # the configured value, used when alternation is off
        # train_act_rows filters the act rows the engine queues; the alternation overwrites it
        # every training iteration (False in the bid phase).
        self.train_act_rows = bool(_get("train_act_rows", True))
        self._alt_act_cfg = self.train_act_rows

        self.bid_levels = tuple(float(x) for x in _get("bid_levels", (0.0, 0.2, 0.4, 0.6, 0.8)))
        # bid_pass: level 0 = pass (sit out, cannot win). Hard-wired in MPT.
        self.bid_pass = bool(_get("bid_pass", True))
        assert self.bid_pass, "MPT requires bid_pass=True"
        assert 0.0 in self.bid_levels, "bid_pass=True needs a 0.0 level to snap passes onto"
        self.seg_scale = bool(_get("seg_scale", self.SEG_SCALE_DEFAULT))
        assert self.reward_rent > 0.0, "revenue_share expects a participation rent (e.g. 0.05)"

        # Fair random tie-breaking. Bids are highly discrete; a plain argmax hands every tie to
        # the lowest-index agent, which starves the others. Jitter is << the bid granularity.
        self._rng = np.random.default_rng(0)

        # decision trace (JSONL, append-only, off unless <TRACE_ENV> is set)
        self._trace_path = os.environ.get(self.TRACE_ENV, "") if self.TRACE_ENV else ""
        self._trace_text = int(os.environ.get(f"{self.TRACE_ENV}_TEXT", "400")) if self.TRACE_ENV else 400
        self._pid = os.getpid()
        self._ep_seq = 0
        self._is_train = True
        # published for the counterfactual collector's pre-action hook (see run())
        self.last_text_actions = None
        if self.cf_preference:
            from agent_system.multi_turn_rollout.counterfactual_collector import CounterfactualCollector
            self.cf_collector = CounterfactualCollector(self, cfg)
        self._init_domain()
        self._reset_episode()

    # ==========================================================================================
    # domain hooks
    # ==========================================================================================
    def _init_domain(self):
        """Extra per-domain settings (trace variables); called once from __init__."""

    def _turn_context(self, env_obs: Dict[str, Any], bs: int):
        """Per-turn context for the trace, read from the observation BEFORE the agents bid."""
        return None

    def _after_turn(self, t, step, bs, active, res, bid_texts, acts, ctx):
        """Trace record(s) and the log line of one auctioned turn."""

    def _after_finalize(self, bundle, bs: int, episode_rewards, is_train: bool):
        """Trace record(s) after the end-of-episode settlement."""

    # ==========================================================================================
    # act/bid alternation
    # ==========================================================================================
    def _maybe_sync_alt_counter(self):
        """Align the alternation counter with the checkpoint the trainer resumes from.

        A crash between two saves leaves the counter up to save_freq-1 iterations ahead of the
        resumed checkpoint, so the 4:1 cycle would restart at a shifted position. Under
        trainer.resume_mode=auto with a checkpoint present, the counter is set to that
        checkpoint's step (N completed iterations -> counter N). No checkpoint: the counter stays
        as it is. MPT_ALT_SYNC=0 (or MC_ALT_SYNC=0) disables the re-sync."""
        if os.environ.get("MPT_ALT_SYNC", os.environ.get("MC_ALT_SYNC", "1")) == "0":
            return
        trainer_cfg = getattr(self.config, "trainer", None)
        if trainer_cfg is None:
            return
        step = checkpoint_iteration(trainer_cfg)
        if step is None:
            return
        print(f"[{self.LOG_TAG}] alternation counter synced to the checkpoint step {step} "
              f"(counter file: {self._alt_counter or '<in-process>'})", flush=True)
        self._alt_machine.sync_to(step)

    def _alt_apply(self):
        """Resolve this iteration's act/bid phase. Mechanism and caveats: see the docstring of
        economy_engine.engine.AlternatingPhase (must be ticked from begin_episode(is_train=True),
        never from __init__ or reset())."""
        if not hasattr(self, "_alt_machine"):
            self._alt_machine = AlternatingPhase(self.alt_act_iters, self.alt_bid_iters,
                                                 self._alt_counter,
                                                 self._alt_bid_cfg, self._alt_act_cfg)
            if self._alt_machine.enabled:
                self._maybe_sync_alt_counter()
        tb, ta, phase = self._alt_machine.apply()
        self.alt_phase = phase
        self.train_bid_rows, self.train_act_rows = tb, ta
        # Counterfactual preference mode maps onto the alternation: a bid-phase iteration
        # becomes a PREFERENCE iteration -- the engine queues NO policy-gradient rows at all
        # (neither bid nor act), and the counterfactual collector supplies the iteration's only
        # training rows (answer-level preference pairs, consumed by dp_actor's
        # bid_pref_scope="answer" branch). Act-phase iterations are untouched. Train-only:
        # _alt_apply is gated to training episodes by begin_episode.
        self.cf_active = bool(getattr(self, "cf_preference", False) and tb)
        if self.cf_active:
            self.train_bid_rows = False
            self.train_act_rows = False
            print(f"[{self.LOG_TAG}] counterfactual preference phase (iter phase={phase}): "
                  f"PG rows OFF, collecting branch pairs instead", flush=True)

    # ==========================================================================================
    # per-episode state
    # ==========================================================================================
    def _reset_episode(self):
        """Per-episode state lives in the framework-agnostic engine; the orchestra keeps
        only what the engine must not know about: batch handles (verl DataProto) and the trace
        record lifecycle."""
        if not hasattr(self, "engine"):
            self.engine = ChainTurnEngine(
                self.n, bid_levels=self.bid_levels, reward_rent=self.reward_rent,
                reward_scale=self.reward_scale, seg_scale=self.seg_scale, rng=self._rng)
        self.engine.begin_episode()
        self._handles: Dict[Any, Any] = {}    # (turn, desk_idx, kind) -> batch handle
        self._ep_seq += 1

    # -- engine state views (tests and the trace layer read these) ----------------------------
    @property
    def _turn(self):
        return self.engine.turn

    @property
    def _prices(self):
        return self.engine.prices

    @property
    def _chosen(self):
        return self.engine.chosen

    @property
    def _bidders(self):
        return self.engine.bidders

    @property
    def _term_at(self):
        return self.engine.term_at

    @property
    def _betas(self):
        return self.engine.betas

    @property
    def _step_rewards(self):
        return self.engine.step_rewards

    @property
    def _held(self):
        return self.engine._held

    @property
    def _bid_fallbacks(self):
        return self.engine.bid_fallbacks

    def reset(self):
        super().reset()          # clears the buffer and every agent
        self._reset_episode()

    def note_step_rewards(self, rewards: np.ndarray):
        """Hook called by rollout_loop right after envs.step(): the per-turn env reward stream
        the recursive settlement needs."""
        self.engine.note_step_rewards(rewards)

    def begin_episode(self, is_train: bool = True):
        """Hook called by rollout_loop right after reset(); labels the trace AND advances the
        act/bid alternation phase."""
        self._is_train = bool(is_train)
        # The alternation phase must advance once per TRAINING rollout batch and never on an
        # evaluation batch (is_train=False), which would otherwise advance the counter and
        # desynchronise the phase from the training iteration. It has to tick here rather than
        # in reset(): rollout_loop calls reset() BEFORE begin_episode(), so a guard inside
        # reset() would read the previous batch's stale flag. With alternation off (alt_*=0)
        # apply() returns "fixed" before ticking, so this is a no-op.
        if self._is_train:
            self._alt_apply()

    # ==========================================================================================
    # trace / worker groups
    # ==========================================================================================
    def _trace(self, rows: List[dict]):
        if not self._trace_path or not rows:
            return
        try:
            with open(self._trace_path, "a") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        except Exception as e:      # a trace write must never kill a rollout
            print(f"[{self.LOG_TAG}] trace write failed (non-fatal): {e}", flush=True)

    def _wg(self, actor_rollout_wgs, agent_id):
        return actor_rollout_wgs[self.agents_to_wg_mapping[agent_id]]

    def _wg_bid(self, actor_rollout_wgs, agent_id):
        """Worker group that answers this agent's BID probe. Normally the agent's own group; an
        evaluation may register `<wg_id>@bid` (other weights for bidding than for acting); the
        trainer never does, so training runs are unchanged."""
        alt = actor_rollout_wgs.get(self.agents_to_wg_mapping[agent_id] + "@bid") \
            if hasattr(actor_rollout_wgs, "get") else None
        return alt if alt is not None else self._wg(actor_rollout_wgs, agent_id)

    # ==========================================================================================
    # emission
    # ==========================================================================================
    # KEY UNIFORMITY IS LOAD-BEARING. gather_rollout_data collates EVERY row of the whole rollout
    # (bid rows and act rows, all agents) with one collate_fn, which stacks per key -- a key that
    # exists on only some rows produces a short stack and DataProto.from_single_dict then dies on
    # the batch-size mismatch. Bid rows carry 'bid', act rows carry
    # 'instruction'/'first_word'/'whitelist_ok', so those must be stripped before the row leaves
    # the orchestra. The values are already copied into the JSONL trace, and 'raw_response' is
    # dropped too because it is a long string per row that nothing downstream reads.
    _STRIP_KEYS = ('bid', 'raw_response', 'instruction', 'first_word', 'whitelist_ok')

    def _stamp(self, batch, mask: np.ndarray, credit: np.ndarray, rent: float, bs: int):
        for k in self._STRIP_KEYS:
            batch.non_tensor_batch.pop(k, None)
        batch.non_tensor_batch['econ_credit_internal'] = np.where(
            mask, credit - rent, 0.0).astype(np.float32)
        # is_final=0 on every row: R is added exactly once, by finalize_episode, into `credit`.
        batch.non_tensor_batch['econ_is_final'] = np.zeros(bs, dtype=np.float32)
        batch.non_tensor_batch['agent_active_mask'] = mask
        return batch

    # ==========================================================================================
    # one env turn
    # ==========================================================================================
    def _publish_cf_candidates(self, res, active: np.ndarray, t: int):
        """Counterfactual preference phase: expose this turn's contested auctions so the
        collector (called from rollout_loop AFTER run() and BEFORE envs.step) can snapshot them
        pre-action. A candidate needs >=2 bidders and a runner-up strictly above the lowest bid
        level -- the driver's ranking flip requires a resample strictly BELOW the runner-up,
        which a bottom-level runner-up can never satisfy."""
        if not getattr(self, "cf_active", False):
            return
        _cands = []
        _min_lv = min(self.bid_levels) if self.bid_levels else None
        _forced = getattr(res, "bid_forced", None)
        for _s in np.nonzero(active)[0]:
            # bid_pass: a turn where everybody passed is reopened at the floor with
            # SYNTHETIC 0.2 bids (res.raw_bids is overwritten). It is not contested --
            # a resample "strictly below the runner-up" is the same pass the winner
            # already wrote, and the pair would compare two answers with equal bids.
            if _forced is not None and bool(_forced[_s]):
                continue
            _w = int(res.winner[_s])
            _bids_s = {i: float(res.raw_bids[i][_s]) for i in range(self.n)
                       if res.bidders[i][_s]}
            if len(_bids_s) < 2 or _w not in _bids_s:
                continue
            _others = sorted((b for i, b in _bids_s.items() if i != _w), reverse=True)
            if _min_lv is not None and _others[0] <= _min_lv + 1e-9:
                continue
            _cands.append({"sample": int(_s), "turn": int(t), "winner_idx": _w,
                           "bids": {self.desk_ids[i]: b for i, b in _bids_s.items()}})
        self.cf_turn_candidates = _cands or None

    def run(self, gen_batch: DataProto, env_obs: Dict[str, Any], actor_rollout_wgs,
            active_masks: np.ndarray, step: int) -> Tuple[List[str], List[Dict]]:
        self.reset_buffer()
        text_actions, team_context, env_obs = self.initialize_context(env_obs)
        active = np.asarray(active_masks).astype(bool)
        bs = len(text_actions)
        t = self._turn

        # the state the agents are bidding on (domain hook; e.g. the Minecraft world state)
        ctx = self._turn_context(env_obs, bs)

        if not active.any():
            # Defensive: rollout_loop breaks on is_done.all(), so this is normally unreachable.
            # If it ever happens there is nothing to auction; finalize_episode emits every held row.
            return text_actions, self.multiagent_batch_buffer

        # ---- adapter callbacks: agent generation + handle stash. The engine never touches a
        # DataProto: it hands (turn, agent, kind) back through emit and we resolve the handle. ----
        acts = {"instr": [None] * bs, "verb": [""] * bs, "ok": [False] * bs, "raw": [""] * bs}
        bid_texts: Dict[int, Any] = {}
        orch = self

        class _CB:
            @staticmethod
            def bid(i, mask):
                aid = orch.desk_ids[i]
                bbatch, bids = orch.agents[aid].bid_call(
                    gen_batch=gen_batch, env_obs=env_obs, team_context=team_context,
                    actor_rollout_wg=orch._wg_bid(actor_rollout_wgs, aid),
                    agent_active_mask=mask, step=step)
                orch._handles[(t, i, "bid")] = bbatch
                bid_texts[i] = bbatch.non_tensor_batch.get('raw_response')
                return bids

            @staticmethod
            def act(i, mask):
                aid = orch.desk_ids[i]
                abatch, emitted = orch.agents[aid].call(
                    gen_batch=gen_batch, env_obs=env_obs, team_context=team_context,
                    actor_rollout_wg=orch._wg(actor_rollout_wgs, aid),
                    agent_active_mask=mask, step=step)
                orch._handles[(t, i, "act")] = abatch
                nb = abatch.non_tensor_batch
                _instr = nb.get('instruction')
                for s_ in np.nonzero(mask)[0]:
                    text_actions[s_] = emitted[s_]
                    acts["instr"][s_] = _instr[s_] if _instr is not None else None
                    acts["verb"][s_] = nb['first_word'][s_]
                    acts["ok"][s_] = bool(nb['whitelist_ok'][s_])
                    acts["raw"][s_] = str(nb['raw_response'][s_])[:orch._trace_text]

            @staticmethod
            def emit(i, kind, turn, mask, credit, rent):
                batch = orch._handles[(turn, i, kind)]
                orch.save_to_buffer(orch.desk_ids[i], orch._stamp(batch, mask, credit, rent, bs))

        res = self.engine.run_turn(active, t, bs, train_bid_rows=self.train_bid_rows,
                                   train_act_rows=self.train_act_rows, cb=_CB)
        self._publish_cf_candidates(res, active, t)
        self._after_turn(t, step, bs, active, res, bid_texts, acts, ctx)
        # published for the counterfactual collector's pre-action hook: branch A replays
        # the mainline winner's OWN emitted action instead of drawing a fresh act sample
        self.last_text_actions = list(text_actions)
        return text_actions, self.multiagent_batch_buffer

    # ==========================================================================================
    # end of episode
    # ==========================================================================================
    def finalize_episode(self, env_obs: Dict[str, Any], episode_rewards, is_train: bool = True):
        """Called ONCE by rollout_loop after the env-step loop. The engine settles everything
        that only the full episode can price; this layer resolves batch handles and writes the
        outcome trace records.

        Returns a list of {'agent_id', 'batch'} exactly like `multiagent_batch_buffer`; the
        batches carry `econ_credit_internal` (already the FULL credit) and `econ_is_final` = 0,
        so the caller's usual `econ_credit = internal + rewards * is_final` reproduces it."""
        out: List[Dict[str, Any]] = []
        if not self._prices:
            return out
        bs = len(self._prices[0])
        orch = self

        class _Sink:
            @staticmethod
            def emit(i, kind, turn, mask, credit, rent):
                batch = orch._handles[(turn, i, kind)]
                out.append({"agent_id": orch.desk_ids[i],
                            "batch": orch._stamp(batch, mask, credit, rent, bs)})

        bundle = self.engine.finalize(episode_rewards, bs, is_train, _Sink)
        self._after_finalize(bundle, bs, episode_rewards, is_train)
        return out

    # ---- shared trace helper of the NaVILA / deep-research hooks ----------------------------
    def _episode_summary_trace(self, bs: int, episode_rewards):
        if not self._trace_path:
            return
        summary = []
        for s in range(bs):
            if self._term_at[s] < 0:
                continue
            er = float(np.asarray(episode_rewards).reshape(-1)[s]) if episode_rewards is not None else None
            summary.append({"kind": "episode", "pid": self._pid, "episode": self._ep_seq,
                            "sample": int(s), "train": self._is_train,
                            "turns": int(self._term_at[s]) + 1, "episode_reward": er,
                            "p0": float(self._prices[0][s])})
        self._trace(summary)
