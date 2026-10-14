import os
from typing import List, Tuple

import numpy as np


class AlternatingPhase:
    """Alternating training: several act-only iterations, then one bid-only iteration.

    Why a file-backed counter: there is exactly ONE orchestra instance per training process
    (MultiAgentTrajectoryCollector builds it once in __init__; every iteration calls reset()),
    but retry loops restart python, so the iteration count must survive the process. The counter
    is read-and-incremented once per apply().

    Why apply() must NEVER be called from __init__: if the phase were resolved at construction it
    would freeze at iter=0 (= act phase) forever and alternation would silently degrade to
    act-only training.

    Where it IS called from: `begin_episode(is_train=...)`, guarded by is_train, in every chain
    orchestra. It used to be called from reset(), but rollout_loop calls reset() for validation
    batches too, so a 230-batch validation pass advanced the counter 230 times and desynchronised
    the phase from the training iteration. reset() cannot host an is_train guard either, because
    rollout_loop calls reset() BEFORE begin_episode() (see vanilla_multi_turn_loop) and would read
    the previous batch's stale flag.
    """

    def __init__(self, act_iters: int, bid_iters: int, counter_path: str,
                 bid_default: bool, act_default: bool):
        self.act_iters = int(act_iters)
        self.bid_iters = int(bid_iters)
        self.counter_path = counter_path or ""
        self.bid_default = bool(bid_default)
        self.act_default = bool(act_default)
        self._mem = 0            # in-process counter, used when the file cannot be written

    def sync_to(self, step: int) -> None:
        """Set the counter to `step` (the iteration the trainer resumes at): the next apply()
        resolves iteration `step`. Writes the counter file when there is one, through the same
        path as _tick; a write failure keeps the in-process count."""
        step = int(step)
        self._mem = step
        if not self.counter_path:
            return
        try:
            with open(self.counter_path, "w") as f:
                f.write(str(step))
        except Exception as e:                                        # noqa: BLE001
            print(f"[alt] counter write failed ({e}); counting in-process (iter={step})", flush=True)

    @property
    def enabled(self) -> bool:
        return self.act_iters > 0 and self.bid_iters > 0

    def apply(self) -> Tuple[bool, bool, str]:
        """Resolve this iteration's phase. Returns (train_bid_rows, train_act_rows, phase)."""
        if not self.enabled:
            return self.bid_default, self.act_default, "fixed"
        k = self._tick()
        period = self.act_iters + self.bid_iters
        in_bid = (k % period) >= self.act_iters
        phase = "bid" if in_bid else "act"
        print(f"[alt] act/bid alternation iter={k} period={period} -> phase={phase} "
              f"(train_bid_rows={in_bid}, train_act_rows={not in_bid})", flush=True)
        return in_bid, (not in_bid), phase

    def _tick(self) -> int:
        """Read-and-increment the persistent counter, never crashing the run.

        A failure must NOT return 0. The old version did, and the read and the write shared one
        try-block: when the scratch filesystem went over quota the write raised EDQUOT, the already-read k was
        thrown away, and every iteration reported iter=0. Since the bid phase is `k % 5 >= 4`,
        iter=0 is always the act phase -- so four arms advertised as alternating act/bid trained
        their bidding head exactly zero times, for hours, announced only by one informational
        log line. Read and write are separate now, and a write failure falls back to counting in
        process so the phase still cycles.
        """
        if not self.counter_path:
            k, self._mem = self._mem, self._mem + 1
            return k
        k = self._mem
        try:
            if os.path.exists(self.counter_path):
                k = int(open(self.counter_path).read().strip() or 0)
        except Exception as e:                                        # noqa: BLE001
            print(f"[alt] counter read failed ({e}); using in-process count iter={k}", flush=True)
        self._mem = k + 1
        try:
            with open(self.counter_path, "w") as f:
                f.write(str(k + 1))
        except Exception as e:                                        # noqa: BLE001
            print(f"[alt] counter write failed ({e}); counting in-process (iter={k}, the phase "
                  f"still cycles)", flush=True)
        return k


def settle_revenue_share(betas: List[float], rewards: List[float],
                         seg_scale: bool = True) -> Tuple[List[float], float, float]:
    """Pure per-sample revenue-share settlement.

    betas[t] = winning share of turn t; rewards[t] = env reward of turn t. Returns
    (profit_per_turn, scaled_total, leak) with conservation
    sum(profit) == scaled_total - leak. seg_scale multiplies each reward by the number of
    turns since the previous reward event (the xN design).
    """
    T = len(betas)
    rhat = {}
    prev_ev = 0
    for t in range(T):
        if rewards[t] > 0:
            n_seg = (t + 1 - prev_ev) if seg_scale else 1
            rhat[t] = rewards[t] * n_seg
            prev_ev = t + 1
    income_next, beta_next = 0.0, 0.0
    profit = [0.0] * T
    for t in range(T - 1, -1, -1):
        inc = rhat.get(t, 0.0) + beta_next * income_next
        profit[t] = (1.0 - betas[t]) * inc
        income_next, beta_next = inc, betas[t]
    return profit, sum(rhat.values()), beta_next * income_next


class TurnResult:
    """Numeric outcome of one auctioned turn, for the caller's trace/logging. Text/batch handles
    are NOT here: the caller created them inside its own callbacks and keeps them itself."""

    __slots__ = ("raw_bids", "bid_vals", "bidders", "winner", "price", "chosen", "win_masks",
                 "bid_forced")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))


class ChainTurnEngine:
    """Turn-mode driver of the MPT economy: ONE call = one sealed-bid auction over one env turn.
    Winner rows are held and settled at the end of the episode by the revenue-share recursion.

    Framework-agnostic by construction: batch objects are opaque handles that flow from the
    caller's callbacks straight into the caller's `emit`; the engine only does masks, prices,
    credits and bookkeeping.

    Callbacks (a duck-typed object):
        bid(i, mask)           -> float[bs]  (the caller keeps its own batch handle)
        act(i, mask)           -> None       (ditto; the caller collects env action texts)
        emit(i, kind, turn, mask, credit, rent)  hand a settled row to the trainer sink;
            the caller resolves its handle by (turn, i, kind)
    """

    def __init__(self, n: int, *, bid_levels, reward_rent: float, reward_scale: float,
                 seg_scale: bool, rng=None):
        self.n = int(n)
        self.bid_levels = tuple(bid_levels)
        # MPT hard-wires revenue-share settlement and bid_pass (level 0 = pass, cannot win);
        # the participation rent is what makes passing on a useless turn the better choice.
        assert 0.0 in self.bid_levels, "MPT uses bid_pass: bid_levels must contain 0.0 (= pass)"
        assert float(reward_rent) > 0.0, "MPT requires a positive participation rent"
        self.reward_rent = float(reward_rent)
        self.reward_scale = float(reward_scale)
        self.seg_scale = bool(seg_scale)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.bid_fallbacks = 0
        # MPT_DESK_PARALLEL=1 (the older name MC_DESK_PARALLEL=1 is accepted too): run each
        # phase's per-agent LLM calls CONCURRENTLY (one python
        # thread per agent; the ray generate calls release the GIL while waiting on their
        # separate sglang engines). Agents within a phase are data-independent -- bids land in
        # per-agent arrays, act winners hold disjoint sample masks -- and every engine-side
        # mutation stays on the main thread (workers only run the cb call and hand the result
        # back), so parallel is result-identical to serial: each engine's sampling RNG advances
        # only with ITS OWN call sequence, which is unchanged. Phase boundaries stay strictly
        # ordered (acts need the cleared auction). Env-gated so serial behavior is the default.
        self.desk_parallel = os.environ.get(
            "MPT_DESK_PARALLEL", os.environ.get("MC_DESK_PARALLEL", "0")) == "1"
        self._desk_pool = None
        self.begin_episode()

    # ---------------- per-episode state ----------------
    def begin_episode(self):
        self.turn = 0
        self.prices: List[np.ndarray] = []
        self.chosen: List[np.ndarray] = []
        self.bidders: List[np.ndarray] = []
        self.term_at: "np.ndarray | None" = None
        self.betas: List[np.ndarray] = []
        self.step_rewards: List[np.ndarray] = []
        self._held: List[dict] = []     # ALL winner rows hold until finalize

    def note_step_rewards(self, rewards):
        """rollout_loop calls this after envs.step(): the per-turn env reward stream that the
        revenue-share recursion settles against."""
        self.step_rewards.append(np.asarray(rewards, dtype=np.float64).reshape(-1))

    # ---------------- one auctioned turn ----------------
    def _per_desk(self, fn, masks: dict):
        """Run one phase's per-agent callback over `masks` ({desk_idx: mask}) and yield
        (desk_idx, result) in key order. Serial by default; with desk_parallel the calls run
        concurrently on one thread per agent while results are still APPLIED in key order on
        the caller's (main) thread, so engine-side state mutation order is deterministic and
        identical to the serial path. Exceptions re-raise on result collection -- same
        failure semantics as the serial loop."""
        if not self.desk_parallel or len(masks) <= 1:
            for i in sorted(masks):
                yield i, fn(i, masks[i])
            return
        if self._desk_pool is None:
            from concurrent.futures import ThreadPoolExecutor
            self._desk_pool = ThreadPoolExecutor(max_workers=self.n,
                                                 thread_name_prefix="agent")
        futs = {i: self._desk_pool.submit(fn, i, masks[i]) for i in sorted(masks)}
        for i in sorted(futs):
            yield i, futs[i].result()

    def run_turn(self, active: np.ndarray, t: int, bs: int, *, train_bid_rows: bool,
                 train_act_rows: bool, cb) -> TurnResult:
        from economy_engine.auction import snap_bids_to_levels, apply_bid_pass, clear_auction
        if self.term_at is None:
            self.term_at = np.full(bs, -1, dtype=np.int64)

        # ---- 1) collect bids from every agent on every active sample (raw quote vs effective
        #         price are separate tracks) ----
        raw_bids = [np.zeros(bs, dtype=np.float64) for _ in range(self.n)]
        bid_vals = np.full((self.n, bs), -np.inf, dtype=np.float64)
        bidders_t = [np.zeros(bs, dtype=bool) for _ in range(self.n)]
        _bid_masks = {i: active for i in range(self.n)} if active.any() else {}
        for i, b in self._per_desk(cb.bid, _bid_masks):
            bid_mask = _bid_masks[i]
            bids = np.asarray(b, dtype=np.float64)
            raw_bids[i][bid_mask] = bids[bid_mask]
            bid_vals[i][bid_mask] = bids[bid_mask]
            bidders_t[i] = bid_mask

        # ---- 1b) snap to the levels; level 0 = pass. bid_forced[s] = every agent passed and
        #          the turn was reopened at the floor with synthetic bids; such a turn is not a
        #          contested auction ----
        snap_bids_to_levels(raw_bids, bid_vals, bidders_t, active, self.bid_levels)
        bid_forced = apply_bid_pass(raw_bids, bid_vals, bidders_t, active, self.bid_levels)
        self.bid_fallbacks += int(bid_forced.sum())

        # ---- 2) clear ----
        winner, price_t, chosen_t = clear_auction(bid_vals, active, self.rng)
        return self._act_and_book(active, t, bs, cb, raw_bids, bid_vals, bidders_t,
                                  bid_forced, winner, price_t, chosen_t,
                                  train_bid_rows, train_act_rows)

    def _act_and_book(self, active, t, bs, cb, raw_bids, bid_vals, bidders_t, bid_forced,
                      winner, price_t, chosen_t, train_bid_rows, train_act_rows):
        # ---- 3) act: the winner emits the env action ----
        win_masks: dict = {}
        for i in range(self.n):
            win_i = active & (winner == i)
            if win_i.any():
                win_masks[i] = win_i
        for _i, _r in self._per_desk(cb.act, win_masks):
            pass                          # act() returns nothing; winners' masks are disjoint

        # ---- 4) book the turn ----
        self._prices_append(price_t, chosen_t, bidders_t, active, t)

        # ---- 5) settle: the pass-back share of turn t is the winning bid ----
        self.betas.append(price_t.copy())
        # losers settle now (credit 0: only the agent that TAKES the turn pays the rent --
        # when both sides paid it, grabbing a useless turn beat standing aside by +0.005 and
        # ten iterations of REINFORCE duly bid it up); winner rows hold to finalize.
        if train_bid_rows:
            for i in range(self.n):
                lose_i = active & bidders_t[i] & (winner != i)
                if lose_i.any():
                    cb.emit(i, "bid", t, lose_i, np.zeros(bs), 0.0)
        for i in win_masks:
            if train_bid_rows:
                self._held.append({"i": i, "kind": "bid", "mask": win_masks[i].copy(),
                                   "turn": t, "rent": self.reward_rent})
            # the act row carries the settled profit either way, so dropping the bid row
            # removes a training sample without touching the economy's arithmetic
            if train_act_rows:
                self._held.append({"i": i, "kind": "act", "mask": win_masks[i].copy(),
                                   "turn": t, "rent": self.reward_rent})
        self.turn = t + 1
        return TurnResult(raw_bids=raw_bids, bid_vals=bid_vals, bidders=bidders_t,
                          winner=winner, price=price_t, chosen=chosen_t, win_masks=win_masks,
                          bid_forced=bid_forced)

    def _prices_append(self, price_t, chosen_t, bidders_t, active, t):
        self.prices.append(price_t)
        self.chosen.append(chosen_t)
        self.bidders.append(bidders_t)
        self.term_at[active] = t

    # ---------------- end of episode ----------------
    def finalize(self, episode_rewards, bs: int, is_train: bool, cb):
        """Settle everything that only the full episode can price. Returns the numeric bundle
        the caller needs for its trace layer:
            {"mode": "rs", "profit": (T,bs), "R_steps", "T", "cons_err"}
        Row emission happens through cb.emit exactly as during turns."""
        if not self.prices:
            return None
        return self._finalize_rs(episode_rewards, bs, cb)

    def _finalize_rs(self, episode_rewards, bs, cb):
        """Recursive revenue-share settlement over each sample's full episode, then emit every
        held winner row: credit = profit(turn) - rent (both the bid row and the act row)."""
        T = len(self.betas)
        R_steps = self.step_rewards
        if len(R_steps) < T:            # defensive: missing note_step_rewards -> zero-fill
            R_steps = R_steps + [np.zeros(bs)] * (T - len(R_steps))
        profit = np.zeros((T, bs))
        cons_err = 0.0
        for s_i in range(bs):
            last = int(self.term_at[s_i]) if self.term_at is not None else T - 1
            if last < 0:
                continue
            b = [float(self.betas[t][s_i]) for t in range(last + 1)]
            r = [float(R_steps[t][s_i]) for t in range(last + 1)]
            p, scaled_tot, leak = settle_revenue_share(b, r, self.seg_scale)
            for t in range(last + 1):
                profit[t, s_i] = p[t]
            cons_err = max(cons_err, abs(sum(p) - (scaled_tot - leak)))
        n_rows = 0
        for row in self._held:
            cred = profit[row["turn"]].astype(np.float64)
            cb.emit(row["i"], row["kind"], row["turn"], row["mask"], cred, row["rent"])
            n_rows += 1
        print(f"[chain/rs] finalize: turns={T} rows={n_rows} "
              f"conservation_maxerr={cons_err:.2e}", flush=True)
        self._held = []
        return {"mode": "rs", "profit": profit, "R_steps": R_steps, "T": T,
                "cons_err": cons_err}
