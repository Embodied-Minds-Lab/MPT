from typing import List, Tuple

import numpy as np


def snap_bids_to_levels(raw_bids: List[np.ndarray], bid_vals: List[np.ndarray],
                        bidders: List[np.ndarray], active: np.ndarray,
                        levels) -> None:
    """Snap every parsed bid to the nearest allowed level, in place.

    raw_bids[i] is agent i's RAW quote per sample (what gets traced / written back into <bid>
    prefixes); bid_vals[i] is the EFFECTIVE price entering the auction (the raw/effective split
    is a two-track design, keep it).
    """
    lv = np.asarray(levels)
    n = len(raw_bids)
    for i in range(n):
        snapped = lv[np.abs(raw_bids[i][None, :] - lv[:, None]).argmin(axis=0)]
        raw_bids[i] = np.where(bidders[i], snapped, 0.0)
        m = active & bidders[i]
        bid_vals[i][m] = snapped[m]


def apply_bid_pass(raw_bids: List[np.ndarray], bid_vals: List[np.ndarray],
                   bidders: List[np.ndarray], active: np.ndarray,
                   levels) -> np.ndarray:
    """Level 0 = pass (sit out, cannot win). Returns the bool[bs] mask of
    SAMPLES whose turn had to reopen at the floor because everyone passed (its sum is the
    caller's fallback counter -- see module docstring).

    The mask is returned rather than its count because a floor-reopened turn is NOT a
    contested auction: every agent actually said "pass", the floor bids are synthetic. The
    counterfactual-preference candidate filter must see this -- on the post-overwrite raw
    bids alone a 0.2/0.2/0.2 reopened turn looks contested, the driver then "flips" the
    winner by resampling a bid strictly below 0.2, i.e. the very same pass the winner had
    already written, and the resulting pair compares two answers with identical bids."""
    n = len(raw_bids)
    for i in range(n):
        bid_vals[i][raw_bids[i] == 0.0] = -np.inf
    floor = float(min(l for l in levels if l > 0))
    dead = active & ~np.isfinite(np.asarray(bid_vals)).any(axis=0)
    if dead.any():
        for i in range(n):
            m = dead & bidders[i]
            bid_vals[i][m] = floor
            raw_bids[i][m] = floor
    return dead


def rank_bids(bid_vals: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The shared tie-break primitive: highest effective bid wins, EXACT ties broken uniformly
    at random by one jitter draw of shape (n, bs).

    A plain argmax hands every tie to the lowest-index agent, which starves the others. Jitter is two orders of
    magnitude below the ~0.1 bid granularity, so it never flips a strict ordering. The caller's
    rng makes exactly ONE draw here: generator consumption order is behavior (same seed, same
    trajectory)."""
    jitter = rng.uniform(0.0, 1e-3, size=bid_vals.shape)
    return np.argmax(np.where(np.isfinite(bid_vals), bid_vals + jitter, -np.inf), axis=0)


def clear_auction(bid_vals: np.ndarray, active: np.ndarray,
                  rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """rank_bids + the turn-mode price extraction. Returns (winner[bs] int, price[bs] float64,
    chosen[bs] int64 with -1 for inactive). `winner` is argmax over agents INCLUDING inactive
    samples (harmless garbage there, exactly as the reference computed it); `chosen` is the
    masked version consumers should use."""
    bs = bid_vals.shape[1]
    winner = rank_bids(bid_vals, rng)
    price = np.zeros(bs, dtype=np.float64)
    price[active] = bid_vals[winner[active], np.nonzero(active)[0]]
    chosen = np.full(bs, -1, dtype=np.int64)
    chosen[active] = winner[active]
    return winner, price, chosen
