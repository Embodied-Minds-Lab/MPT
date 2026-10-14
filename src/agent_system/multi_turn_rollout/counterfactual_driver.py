import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class ContestedTurn:
    """One contested auction turn, as seen at the branch point.

    Attributes:
        bids: agent id -> parsed bid value for every agent that bid this turn.
        winner_response: the first-place agent's full bid response text.
        context: the first-place agent's generation context (whatever ``regenerate``
            expects — prompt, message list, ...); also returned in ``Pair.context``.
        winner: first-place agent id; defaults to the argmax of ``bids`` (pass it
            explicitly when upstream tie-breaking picked a non-argmax winner).
        responses: optional full response text of other agents (the runner-up's is
            needed for branch B; if absent it is regenerated from ``contexts``, and a
            regenerated response is only accepted when its parsed bid reproduces the
            original runner-up bid — otherwise no pair).
        contexts: optional per-agent generation contexts (only consulted when the
            runner-up's response is missing from ``responses``).
    """

    bids: Dict[str, float]
    winner_response: str
    context: Any
    winner: Optional[str] = None
    responses: Dict[str, str] = field(default_factory=dict)
    contexts: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Pair:
    """An answer-level preference pair: same agent, same context, two full answers."""

    y_w_text: str
    y_l_text: str
    context: Any
    meta: Dict[str, Any]


def _restore_failed(result: Any, require_server_state: bool = True) -> bool:
    """Detectable restore-failure signature (exceptions are caught separately).

    FAIL-CLOSED: the adapter contract is "return the per-slot reply DICT". Anything else —
    ``None``, a list, a raw HTTP envelope — is treated as a FAILED restore, never waved
    through: an unrecognized shape proves the adapter is not looking at the slot's actual
    reply, and a branch scored on an unverified restore is worse than no branch. The one
    convenience unwrap: the MC ``/restore`` endpoint's whole-body envelope
    ``{"results": [<one reply>]}`` is unwrapped when it holds exactly one item (the driver
    restores one slot at a time); a multi-item envelope is ambiguous and fails.

    ``require_server_state`` (default True): an MC-style restore reply — recognized by the
    ``blocks``/``cmds`` keys only that server emits — that lacks ``"server_state": true``
    means world+player were replayed but the Python-side scoring state was NOT applied
    (legacy snapshot, MC_SNAPSHOT unset in the restoring process, a seed mismatch, or a
    failed counter rebase). The branch would then run with the TARGET slot's reward_memory —
    already-scored milestones pay zero, the exact bug the full-state snapshot exists to
    kill — so treat it as a failure. Dict replies without those keys (e.g. navila's
    step-shaped ``{"obs": {...}}`` payload) never carry the field and pass this check.
    """
    if result is False:
        return True
    if isinstance(result, dict) and isinstance(result.get("results"), list):
        if len(result["results"]) != 1:
            return True
        result = result["results"][0]
    if not isinstance(result, dict):
        return True                     # None / list / Response object / anything opaque
    if result.get("ok") is False or result.get("success") is False:
        return True
    if "error" in result and result["error"]:
        return True
    if (
        require_server_state
        and ("blocks" in result or "cmds" in result)
        and result.get("server_state") is not True
    ):
        return True
    return False


class CounterfactualBranchDriver:
    """Reusable driver for the resample-based branching protocol (see module docstring).

    Args:
        regenerate: ``(agent, context) -> str``, sampled full bid response.
        parse_bid: ``(response_text) -> Optional[float]``.
        act_fn: ``(env, agent, response_text) -> obs``.
        policy_step: ``(env, obs) -> obs``, one further turn under the same policy.
        score_fn: ``(before_state, after_obs) -> float``.
        k: max resamples of the first-place agent's response (default 8).
        p: further turns each branch rolls out after the branch action (default 5).
        log_fn: sink for progress/failure lines (default: module logger at INFO).
    """

    def __init__(
        self,
        regenerate: Callable[[Any, Any], str],
        parse_bid: Callable[[str], Optional[float]],
        act_fn: Callable[[Any, Any, str], Any],
        policy_step: Callable[[Any, Any], Any],
        score_fn: Callable[[Any, Any], float],
        k: int = 8,
        p: int = 5,
        bid_levels: Optional[list] = None,
        log_fn: Optional[Callable[[str], None]] = None,
    ):
        self.regenerate = regenerate
        self.parse_bid = parse_bid
        self.act_fn = act_fn
        self.policy_step = policy_step
        self.score_fn = score_fn
        self.k = int(k)
        self.p = int(p)
        # bid_levels (required): the economy's discrete bid grid ([0, 0.2, 0.4, 0.6, 0.8] in
        # MPT). Every bid that enters a comparison -- the recorded turn bids, resampled
        # candidates, regenerated runner-up stand-ins -- is snapped to the nearest level
        # first. The auction itself ranks SNAPPED bids (economy_engine snap_bids_to_levels),
        # so comparing raw parsed floats against recorded values would let a formatting
        # wobble (0.6000001) flip the strict "<" test or make the regen equality gate
        # unsatisfiable.
        if not bid_levels:
            raise ValueError("CounterfactualBranchDriver: bid_levels (the auction's bid grid) is required")
        self.bid_levels = sorted(float(b) for b in bid_levels)
        self.log = log_fn if log_fn is not None else logger.info
        # Meta of the most recent build_pair attempt, kept even when None is returned,
        # so failed attempts are never silent.
        self.last_meta: Dict[str, Any] = {}

    def _snap(self, bid: Optional[float]) -> Optional[float]:
        """Nearest bid level (None stays None)."""
        if bid is None:
            return bid
        return min(self.bid_levels, key=lambda lv: abs(lv - float(bid)))

    # ------------------------------------------------------------------ helpers

    def _pick_desks(self, turn: ContestedTurn):
        if len(turn.bids) < 2:
            raise ValueError(f"CounterfactualBranchDriver: need >=2 bids, got {turn.bids}")
        winner = turn.winner if turn.winner is not None else max(turn.bids, key=turn.bids.get)
        if winner not in turn.bids:
            raise ValueError(f"CounterfactualBranchDriver: winner {winner!r} not in bids {turn.bids}")
        others = {d: b for d, b in turn.bids.items() if d != winner}
        runner_up = max(others, key=others.get)
        return winner, runner_up, others[runner_up]

    def _rollout_branch(self, env: Any, desk: Any, response_text: str, meta: Dict[str, Any], tag: str):
        """First action + p policy turns on one branch; returns final obs or None on error."""
        try:
            obs = self.act_fn(env, desk, response_text)
            turns_done = 0
            for _ in range(self.p):
                obs = self.policy_step(env, obs)
                turns_done += 1
            meta[f"branch_{tag}_turns"] = turns_done
            return obs
        except Exception as e:  # noqa: BLE001 - branch failures must be recorded, not raised
            meta[f"branch_{tag}_error"] = f"{type(e).__name__}: {e}"
            self.log(f"[CFDriver] branch {tag} rollout failed: {type(e).__name__}: {e}")
            return None

    # ------------------------------------------------------------------ protocol

    def build_pair(self, turn: ContestedTurn, snapshot_state: Any, env_a: Any, env_b: Any) -> Optional[Pair]:
        """Run the full protocol on one contested turn; ``None`` when no pair is produced.

        Args:
            turn: the contested turn as seen at the branch point.
            snapshot_state: environment state captured by the CALLER at the branch
                point, strictly BEFORE the winner's action was executed anywhere. Both
                branches start from a ``restore(snapshot_state)`` — the driver never
                assumes any live slot is still sitting at the pre-action point, and
                never steps the mainline slot.
            env_a: spare slot for branch A (original winner acts its original response).
            env_b: spare slot for branch B (runner-up acts). Must be a different slot
                from ``env_a``; pass the live mainline slot as neither.
        """
        # A precondition raise below must not leave the PREVIOUS attempt's meta in
        # last_meta -- a reused driver would then report the wrong turn's diagnostics.
        self.last_meta = {}
        if env_a is env_b:
            raise ValueError(
                "CounterfactualBranchDriver: env_a and env_b are the same slot -- branch B's "
                "restore would wipe branch A's rollout. Pass two distinct spare slots."
            )
        winner, runner_up, runner_up_bid = self._pick_desks(turn)
        # Comparisons run on the SNAPPED grid (see
        # __init__): the auction ranked snapped values, so the flip test and the regen
        # equality gate must too.
        runner_up_bid = self._snap(runner_up_bid)
        original_bid = self._snap(turn.bids[winner])
        meta: Dict[str, Any] = {
            "desk": winner,
            "runner_up": runner_up,
            "original_bid": original_bid,
            "runner_up_bid": runner_up_bid,
            "k": self.k,
            "p": self.p,
            "resamples_used": 0,
            "resample_parse_failures": 0,
            "resampled_bids": [],
        }
        self.last_meta = meta

        # 1. Resample the first-place agent until the ranking flips.
        candidate_text = None
        candidate_bid = None
        for i in range(self.k):
            text = self.regenerate(winner, turn.context)
            meta["resamples_used"] = i + 1
            bid = self._snap(self.parse_bid(text))
            if bid is None:
                meta["resample_parse_failures"] += 1
                continue
            meta["resampled_bids"].append(bid)
            if bid < runner_up_bid:  # strictly drops below the runner-up -> ranking changes
                candidate_text, candidate_bid = text, bid
                break
        if candidate_text is None:
            meta["outcome"] = "no_ranking_change"
            self.log(
                f"[CFDriver] agent={winner}: no ranking-changing resample in {self.k} tries "
                f"(parse failures={meta['resample_parse_failures']}, bids={meta['resampled_bids']})"
            )
            return None
        meta["candidate_bid"] = candidate_bid

        # 2. Resolve branch B's runner-up response BEFORE spending any env work.
        #    Branch B's premise is that the runner-up won the turn WITH the bid recorded
        #    in ``turn.bids`` — so a regenerated stand-in (temperature > 0) is only
        #    accepted when its parsed bid reproduces that original bid; a different bid
        #    would make the counterfactual incoherent. Retry up to k, else no pair.
        ru_response = turn.responses.get(runner_up)
        if ru_response is not None:
            meta["runner_up_response_source"] = "stored"
        else:
            if runner_up not in turn.contexts:
                raise ValueError(
                    f"CounterfactualBranchDriver: runner-up {runner_up!r} has neither a stored "
                    "response nor a context to regenerate from — pass one of the two."
                )
            meta["runner_up_regen_bids"] = []
            for j in range(self.k):
                text = self.regenerate(runner_up, turn.contexts[runner_up])
                meta["runner_up_regen_attempts"] = j + 1
                # snapped equality: on a discrete grid this is level-equality, which the
                # regeneration can actually satisfy; raw float == would demand bit-identical
                # parses and make this path succeed essentially never
                rb = self._snap(self.parse_bid(text))
                meta["runner_up_regen_bids"].append(rb)
                if rb is not None and rb == runner_up_bid:
                    ru_response = text
                    break
            if ru_response is None:
                meta["outcome"] = "runner_up_regen_bid_mismatch"
                self.log(
                    f"[CFDriver] agent={winner}: runner-up {runner_up!r} regeneration never reproduced "
                    f"the original bid {runner_up_bid} in {self.k} tries "
                    f"(bids={meta['runner_up_regen_bids']}), no pair"
                )
                return None
            meta["runner_up_response_source"] = "regenerated"

        # 3. Branch A: restore the caller's pre-action snapshot onto a spare slot,
        #    original winner acts with its original response. The mainline slot is
        #    never touched, so the winner's action is never double-executed on it.
        try:
            restore_a = env_a.restore(snapshot_state)
        except Exception as e:  # noqa: BLE001
            restore_a = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        meta["restore_a_ok"] = not _restore_failed(restore_a)
        if not meta["restore_a_ok"]:
            meta["outcome"] = "restore_failed"
            meta["restore_a_result"] = repr(restore_a)[:500]
            self.log(f"[CFDriver] restore onto branch-A slot failed: {repr(restore_a)[:500]}")
            return None
        obs_a = self._rollout_branch(env_a, winner, turn.winner_response, meta, "a")
        if obs_a is None:
            # short-circuit: branch B's restore + (1+p)-turn rollout would be discarded
            # unconditionally once A is dead, so do not spend the env/GPU work
            meta["outcome"] = "branch_rollout_failed"
            return None

        # Branch B: same snapshot onto the other spare slot, runner-up acts.
        try:
            restore_b = env_b.restore(snapshot_state)
        except Exception as e:  # noqa: BLE001
            restore_b = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        meta["restore_b_ok"] = not _restore_failed(restore_b)
        if not meta["restore_b_ok"]:
            meta["outcome"] = "restore_failed"
            meta["restore_b_result"] = repr(restore_b)[:500]
            self.log(f"[CFDriver] restore onto branch-B slot failed: {repr(restore_b)[:500]}")
            return None
        obs_b = self._rollout_branch(env_b, runner_up, ru_response, meta, "b")

        if obs_b is None:
            meta["outcome"] = "branch_rollout_failed"
            return None

        # 4. Score both branches on state deltas from the caller's pre-action snapshot
        #    (score_fn may equally read the reward channel — the snapshot carries the
        #    full server state).
        score_a = float(self.score_fn(snapshot_state, obs_a))
        score_b = float(self.score_fn(snapshot_state, obs_b))
        meta["score_a"] = score_a
        meta["score_b"] = score_b
        if score_a == score_b:
            meta["outcome"] = "tie"
            self.log(f"[CFDriver] agent={winner}: branch tie at score {score_a}, no pair")
            return None

        # 5. Higher-scoring branch's first-agent response wins; both texts belong to the
        #    same agent (the first-place agent) for the same context.
        if score_a > score_b:
            y_w_text, y_l_text = turn.winner_response, candidate_text
            meta["chosen_branch"] = "a"
        else:
            y_w_text, y_l_text = candidate_text, turn.winner_response
            meta["chosen_branch"] = "b"
        meta["outcome"] = "pair"
        self.log(
            f"[CFDriver] agent={winner}: pair built after {meta['resamples_used']} resamples, "
            f"scores a={score_a} b={score_b}, chosen={meta['chosen_branch']}"
        )
        return Pair(y_w_text=y_w_text, y_l_text=y_l_text, context=turn.context, meta=meta)
