import json
import os
import time

import numpy as np
import torch

from agent_system.multi_turn_rollout.counterfactual_driver import (
    ContestedTurn,
    CounterfactualBranchDriver,
)
from agent_system.multi_turn_rollout.utils import to_list_of_dict
from economy_engine.auction import apply_bid_pass, rank_bids, snap_bids_to_levels


def _decode_image(b64):
    if not b64:
        return None
    try:
        import base64 as _b64
        import io as _io

        from PIL import Image
        return Image.open(_io.BytesIO(_b64.b64decode(b64))).convert("RGB")
    except Exception:
        return None


def _encode_image_b64(img):
    """PIL image / HWC uint8 array -> JPEG base64 (None on anything else). Inspection-only."""
    try:
        import base64 as _b64
        import io as _io

        from PIL import Image
        if img is None:
            return None
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype(np.uint8))
        if not isinstance(img, Image.Image):
            return None
        buf = _io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=85)
        return _b64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


class _BranchSlot:
    """Driver-facing handle for one spare env slot: restore + accumulated branch reward."""

    def __init__(self, collector, slot_idx):
        self._c = collector
        self.slot = int(slot_idx)
        self.total_reward = 0.0
        self.done = False
        self.cur_obs = None          # bs-wide env_obs dict positioned at this branch's sample
        self.traj = []               # inspection trail: one dict per branch turn (see _do_act)
        self.n_acts = 0              # env steps taken since restore (server turn = snap turn + n_acts)

    def restore(self, snap):
        self.total_reward = 0.0
        self.done = False
        self.cur_obs = None
        self.traj = []
        self.n_acts = 0
        return self._c._env_restore(self.slot, snap)


class CounterfactualCollector:
    """Builds answer-level preference pairs from counterfactual branches (see module doc)."""

    def __init__(self, orchestra, cfg):
        def _get(k, d):
            try:
                v = getattr(cfg, k, None)
                if v is None and isinstance(cfg, dict):
                    v = cfg.get(k, None)
                return d if v is None else v
            except Exception:
                return d

        self.orch = orchestra
        self.max_pairs = int(_get("cf_max_pairs", 4))
        self.k = int(_get("cf_k", 6))
        self.p = int(_get("cf_p", 4))
        # Branch-point selection: the earliest cf_max_pairs contested turns of the iteration
        # fill the budget. Branch scoring: accumulated env reward over the branch window.
        # random stream for the branch auction's tie-break jitter
        self._branch_rng = np.random.default_rng()
        # per-call timeout handed to the env client (HTTP clients; unused by in-process envs)
        self.timeout = float(_get("cf_http_timeout", 1200.0))
        self._client = None           # the env client, bound lazily on first use
        self._spare = None            # [slot_a, slot_b]
        self._pending = []            # snapshot bundles collected this iteration
        self._pair_seq = 0
        self.last_stats = {}
        # Inspection dump (MPT_CF_DUMP=<path>.jsonl; the older MC_CF_DUMP is accepted too): one
        # record per branch-point attempt -- contested-turn frame, both candidate answers in
        # full, each branch's turn-by-turn actions/rewards/frames, scores, and the failure
        # reason when no pair was built.
        # Purely observational: dump errors are swallowed, training never depends on it.
        self.dump_path = os.environ.get("MPT_CF_DUMP") or os.environ.get("MC_CF_DUMP", "")
        self._iter_no = 0

    def _dump(self, rec):
        if not self.dump_path:
            return
        try:
            # pid: _iter_no is an in-process counter that restarts at 1 after a process
            # restart, so a replayed iteration is otherwise indistinguishable from the one
            # it replaces in this append-only file
            rec = {"ts": time.time(), "pid": os.getpid(), "iter": self._iter_no, **rec}
            with open(self.dump_path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        except Exception as e:  # noqa: BLE001 -- inspection must never hurt training
            print(f"[cf-collector] dump write failed: {e}", flush=True)

    # ------------------------------------------------------------------ env plumbing

    def _bind_env(self, env_manager):
        """Bind the env client and its two spare slots (see the module's env adapter
        contract). Loud on every violation: a preference iteration without usable spare slots
        must fail visibly, not degrade into an ordinary no-op iteration that trains nothing."""
        if self._spare is not None:
            return
        client = getattr(env_manager, "envs", None)
        if not hasattr(client, "cf_bind"):
            raise RuntimeError(
                f"cf collector: the env client {type(client).__name__} implements no cf_bind() "
                "(the counterfactual env adapter: cf_bind / cf_snapshot / cf_restore / "
                "cf_step_slot)")
        spare = list(client.cf_bind(timeout=self.timeout))
        if len(spare) != 2 or spare[0] == spare[1]:
            raise RuntimeError(f"cf collector: cf_bind() must return two distinct spare slots, got {spare}")
        self._client = client
        self._spare = spare

    def _env_snapshot(self, sample):
        return self._client.cf_snapshot(int(sample))

    def _env_restore(self, slot, snap):
        return self._client.cf_restore(int(slot), snap)

    def _env_step_slot(self, slot, action):
        return self._client.cf_step_slot(int(slot), action)

    # ------------------------------------------------------------------ per-iteration hooks

    def begin_iteration(self):
        self._pending = []
        self._pair_seq = 0
        self._iter_no += 1
        self.last_stats = {"candidates_seen": 0, "snapshots": 0, "pairs": 0,
                           "no_ranking_change": 0, "restore_failed": 0, "tie": 0,
                           "branch_rollout_failed": 0, "other_no_pair": 0}

    def observe_turn(self, env_manager, env_obs, step):
        """Pre-action hook: snapshot this turn's contested samples (budget permitting).

        Must run AFTER orchestra.run (winners exist) and BEFORE envs.step (the winner's
        action has not executed anywhere -- the driver's snapshot_state contract).
        """
        cands = getattr(self.orch, "cf_turn_candidates", None)
        if not cands:
            return
        self._bind_env(env_manager)
        self.last_stats["candidates_seen"] += len(cands)
        for c in cands:
            if len(self._pending) >= self.max_pairs:
                break
            # a transient HTTP error here would propagate through the mainline rollout loop
            # and abort the whole iteration mid-episode; every other branch-phase call is
            # already guarded, this one was not
            try:
                snap = self._env_snapshot(int(c["sample"]))
            except Exception as e:  # noqa: BLE001
                snap = {"error": f"{type(e).__name__}: {e}"}
            if not isinstance(snap, dict) or snap.get("error"):
                print(f"[cf-collector] snapshot of slot {c['sample']} failed: "
                      f"{str(snap)[:200]} -- skipping candidate", flush=True)
                continue
            # winner_act_text: the mainline's own emitted action for this turn, so branch A
            # can replay it instead of re-sampling the winner's act (see _act_fn)
            _acts = getattr(self.orch, "last_text_actions", None)
            _wact = None
            try:
                if _acts is not None:
                    _wact = str(_acts[int(c["sample"])])
            except Exception:  # noqa: BLE001
                _wact = None
            bundle = {**c, "snap": snap, "env_obs": env_obs, "env_step": step,
                      "winner_act_text": _wact}
            self._pending.append(bundle)
            self.last_stats["snapshots"] += 1
        # one bundle per (sample, turn) is enough; the orchestra refreshes this next turn
        self.orch.cf_turn_candidates = None

    # ------------------------------------------------------------------ generation adapters

    def _one_hot(self, bs, s):
        m = np.zeros(bs, dtype=bool)
        m[s] = True
        return m

    def _bid_once(self, agent_id, env_obs, gen_batch, wgs, s, step):
        """One bid_call restricted to sample ``s``; returns (row_dict, text, parsed_bid).

        parsed_bid is None when the answer carried no parsable ``<bid>`` tag. That case must
        NOT reach the driver as a number: parse_open_bid's failure direction is 0.0, which
        under bid_pass is a legal pass and therefore strictly below every positive runner-up,
        so a resample truncated inside its ``<think>`` would be accepted as the ranking flip
        and trained on as a deliberate pass. bid_call already publishes the parse verdict as
        ``is_action_valid`` (every domain's agents do), so the driver's parse-failure path (which
        keeps resampling and counts resample_parse_failures) becomes reachable.
        """
        orch = self.orch
        bs = len(env_obs["text"])
        batch, bids = orch.agents[agent_id].bid_call(
            gen_batch=gen_batch, env_obs=env_obs, team_context=[""] * bs,
            actor_rollout_wg=orch._wg_bid(wgs, agent_id),
            agent_active_mask=self._one_hot(bs, s), step=step)
        row = to_list_of_dict(batch)[s]
        text = str(batch.non_tensor_batch["raw_response"][s])
        ok = batch.non_tensor_batch.get("is_action_valid")
        bid = float(bids[s]) if (ok is None or bool(ok[s])) else None
        return row, text, bid

    def _act_once(self, agent_id, env_obs, gen_batch, wgs, s, step):
        """One act call restricted to sample ``s``; returns the emitted env action text."""
        orch = self.orch
        bs = len(env_obs["text"])
        _batch, emitted = orch.agents[agent_id].call(
            gen_batch=gen_batch, env_obs=env_obs, team_context=[""] * bs,
            actor_rollout_wg=orch._wg(wgs, agent_id),
            agent_active_mask=self._one_hot(bs, s), step=step)
        return emitted[s]

    @staticmethod
    def _episode_horizon(env_manager):
        """The mainline episode horizon (``config.env.max_steps``); 0 when the manager carries
        no config (unit-test stubs), which disables the branch-side cap."""
        try:
            return int(env_manager.config.env.max_steps)
        except (AttributeError, TypeError, ValueError):
            return 0

    def _payload_to_obs(self, env_manager, base_obs, s, payload):
        """Fold one branch step payload into a bs-wide env_obs positioned at sample ``s``.

        Delegates to the env manager's ``cf_payload_to_obs`` (the env adapter). The fallback
        below (text through ``build_text_obs`` + one ``image_b64`` frame) serves managers
        without the method. Only position ``s`` is ever consumed (every generation call masks
        to it); the other positions keep the base iteration's content so prompt building stays
        shape-happy.
        """
        _fold = getattr(env_manager, "cf_payload_to_obs", None)
        if _fold is not None:
            return _fold(base_obs, s, payload)
        raw = list(base_obs.get("anchor") or base_obs["text"])
        raw[s] = payload.get("text", "")
        text = env_manager.build_text_obs(raw)
        images = list(base_obs.get("image") or [None] * len(raw))
        im = _decode_image(payload.get("image_b64"))
        if im is not None:
            images[s] = im
        return {"text": text, "image": images, "anchor": raw}

    # ------------------------------------------------------------------ the branch phase

    def build_pairs(self, env_manager, gen_batch, actor_rollout_wgs):
        """Run the driver over every stored branch point; return injectable row bundles.

        Returns a list of ``(sample_idx, [row_dict_w, row_dict_l], driver_meta)``. Rows are
        slices of real generation batches; the caller stamps identity keys (uid/traj_uid/...)
        and appends both rows ADJACENTLY to ``total_batch_list[sample_idx]``.
        """
        out = []
        if not self._pending:
            return out
        orch = self.orch
        regen_rows = {}
        horizon = self._episode_horizon(env_manager)

        # Process bundles grouped by sample: the Minecraft spare slots' build-on-demand restore
        # only rebuilds a world when the snapshot's build seed CHANGES, so consecutive bundles
        # from the same sample reuse the standing branch worlds (~1-2 min world gen saved
        # per avoided rebuild). Order within the driver protocol is otherwise irrelevant.
        for bundle in sorted(self._pending, key=lambda b: int(b["sample"])):
            s = int(bundle["sample"])
            t = int(bundle["turn"])
            w_i = int(bundle["winner_idx"])
            winner_id = orch.desk_ids[w_i]
            env_obs_t = bundle["env_obs"]
            step_t = int(bundle["env_step"])

            handle = orch._handles.get((t, w_i, "bid"))
            if handle is None:
                print(f"[cf-collector] no bid handle for turn {t} agent {winner_id} -- skipping",
                      flush=True)
                self.last_stats["other_no_pair"] += 1
                continue
            winner_row = to_list_of_dict(handle)[s]
            winner_text = str(handle.non_tensor_batch["raw_response"][s])

            # stored runner-up responses: every agent bid this turn, so its full bid answer is
            # in its own stashed handle -- the driver then never needs the regeneration gate
            responses = {}
            for i in range(orch.n):
                h = orch._handles.get((t, i, "bid"))
                if h is not None:
                    responses[orch.desk_ids[i]] = str(h.non_tensor_batch["raw_response"][s])

            def _regen(desk_id, _ctx, _s=s, _obs=env_obs_t, _step=step_t):
                row, text, bid = self._bid_once(desk_id, _obs, gen_batch, actor_rollout_wgs,
                                                _s, _step)
                regen_rows[text] = row
                self._last_bid_by_text[text] = bid
                return text

            self._last_bid_by_text = {winner_text: float(bundle["bids"][winner_id])}
            for d_id, b in bundle["bids"].items():
                if d_id in responses:
                    self._last_bid_by_text.setdefault(responses[d_id], float(b))

            def _parse(text):
                return self._last_bid_by_text.get(text)

            def _do_act(env, desk_id, obs, replay_text=None):
                # replay_text: branch A's first turn re-uses the mainline winner's OWN emitted
                # action instead of drawing a fresh sample, so branch A IS the mainline
                # continuation of turn t and the pair differs in the bid alone (a fresh act
                # sample added a second random variable on top of the p stochastic turns);
                # it also saves one generation call per branch point.
                emitted = replay_text if replay_text is not None else self._act_once(
                    desk_id, obs, gen_batch, actor_rollout_wgs, env._sample, env._step_no)
                payload = self._env_step_slot(env.slot, emitted)
                # A dead or degraded env step must FAIL the branch, never read as "this branch
                # made no further progress": the env servers answer an exception with
                # {"text": "STEP ERROR: ..." / "SLOT ERROR: ...", "reward": 0, "done": True, ...}
                # (Minecraft also sets info.error). Scored silently, it would turn an engine
                # hiccup into a branch preference. The env clients' cf_step_slot raise on their
                # own failure signatures too; this is the generic backstop.
                _err = (payload.get("info") or {}).get("error")
                if _err or str(payload.get("text", "")).startswith(("STEP ERROR", "SLOT ERROR")):
                    raise RuntimeError(f"branch env step failed on slot {env.slot}: "
                                       f"{_err or str(payload.get('text'))[:200]}")
                env.total_reward += float(payload.get("reward", 0.0))
                env.done = bool(payload.get("done", False))
                env._step_no += 1
                # Episode horizon: the mainline stops after env.max_steps turns (the rollout
                # loop's cap), so a branch must stop there too. _step_no is the 1-based turn
                # number of the NEXT turn; once it passes the horizon no further turn exists.
                # The Minecraft / NaVILA servers also end episodes on their own counters; the
                # in-process deep-research env has no horizon of its own and relies on this.
                if horizon > 0 and env._step_no > horizon:
                    env.done = True
                env.cur_obs = self._payload_to_obs(env_manager, obs, env._sample, payload)
                # inspection trail: what actually happened in this branch, turn by turn
                # (frame kept only when dumping -- it is the whole value of the trail)
                _pinfo = payload.get("info") or {}
                env.traj.append({"desk": desk_id, "action": emitted,
                                 "reward": float(payload.get("reward", 0.0)),
                                 "done": env.done,
                                 "inventory": (_pinfo.get("inventory")
                                               if self.dump_path else None),
                                 "frame_b64": (payload.get("image_b64")
                                               if self.dump_path else None)})
                return {"env": env}

            def _act_fn(env, desk_id, _bid_text, _s=s, _obs=env_obs_t, _t=step_t,
                        _replay=bundle.get("winner_act_text")):
                # branch turn 1: the branch slot sits exactly at the mainline's turn-t state,
                # so the act prompt context IS the stored turn-t observation. Branch A (the
                # mainline winner acting) replays the mainline's own emitted action when the
                # orchestra published it; branch B's runner-up never acted on the mainline, so
                # it always generates.
                env._sample = _s
                env._step_no = _t
                env.cur_obs = _obs
                _r = _replay if desk_id == winner_id else None
                return _do_act(env, desk_id, _obs, replay_text=_r)

            def _policy_step(env, state):
                if env.done:
                    return state          # episode over: remaining branch turns are no-ops
                obs = env.cur_obs
                quotes = [self._bid_once(orch.desk_ids[i], obs, gen_batch, actor_rollout_wgs,
                                         env._sample, env._step_no)[2]
                          for i in range(orch.n)]
                return _do_act(env, orch.desk_ids[self.branch_winner(quotes)], obs)

            def _score(_before, state):
                # branch score = env reward accumulated over the branch window
                return float(state["env"].total_reward)

            driver = CounterfactualBranchDriver(
                regenerate=_regen, parse_bid=_parse, act_fn=_act_fn,
                policy_step=_policy_step, score_fn=_score,
                k=self.k, p=self.p, bid_levels=list(orch.bid_levels),
                log_fn=lambda m: print(m, flush=True))

            turn = ContestedTurn(bids=dict(bundle["bids"]), winner_response=winner_text,
                                 context={"sample": s, "turn": t}, winner=winner_id,
                                 responses=responses)
            env_a = _BranchSlot(self, self._spare[0])
            env_b = _BranchSlot(self, self._spare[1])
            # inspection record for this branch-point attempt (dumped on EVERY exit path:
            # the failure distribution is as diagnostic as the successes)
            _rec = None
            if self.dump_path:
                try:
                    _imgs = bundle["env_obs"].get("image") or []
                    _anchor = bundle["env_obs"].get("anchor") or bundle["env_obs"].get("text") or []
                    _rec = {"sample": s, "turn": t, "desk": winner_id,
                            "bids": {k: float(v) for k, v in bundle["bids"].items()},
                            "winner_text": winner_text,
                            "responses": dict(responses),
                            "obs_text": str(_anchor[s]) if s < len(_anchor) else "",
                            "obs_frame_b64": _encode_image_b64(_imgs[s]) if s < len(_imgs) else None}
                except Exception as e:  # noqa: BLE001
                    print(f"[cf-collector] dump record build failed: {e}", flush=True)
                    _rec = {"sample": s, "turn": t, "desk": winner_id}
            try:
                pair = driver.build_pair(turn, bundle["snap"], env_a, env_b)
            except Exception as e:  # noqa: BLE001 - one bad branch point must not kill the batch
                print(f"[cf-collector] driver raised on sample {s} turn {t}: "
                      f"{type(e).__name__}: {e}", flush=True)
                self.last_stats["other_no_pair"] += 1
                if _rec is not None:
                    self._dump({**_rec, "outcome": "driver_exception",
                                "error": f"{type(e).__name__}: {e}",
                                "meta": dict(driver.last_meta),
                                "traj_a": env_a.traj, "traj_b": env_b.traj})
                continue
            outcome = driver.last_meta.get("outcome", "unknown")
            if pair is None:
                self.last_stats[outcome if outcome in self.last_stats else "other_no_pair"] += 1
                if _rec is not None:
                    self._dump({**_rec, "outcome": outcome, "meta": dict(driver.last_meta),
                                "traj_a": env_a.traj, "traj_b": env_b.traj})
                continue

            row_w = winner_row if pair.y_w_text == winner_text else regen_rows.get(pair.y_w_text)
            row_l = winner_row if pair.y_l_text == winner_text else regen_rows.get(pair.y_l_text)
            if row_w is None or row_l is None:
                print(f"[cf-collector] pair text not found among generated rows -- skipping",
                      flush=True)
                self.last_stats["other_no_pair"] += 1
                continue
            weight = abs(float(driver.last_meta["score_a"]) - float(driver.last_meta["score_b"]))
            pid = self._pair_seq
            self._pair_seq += 1
            rows = []
            for row, role in ((dict(row_w), 1), (dict(row_l), -1)):
                row["pref_pair_id"] = torch.tensor(pid, dtype=torch.long)
                row["pref_pair_role"] = torch.tensor(role, dtype=torch.long)
                row["pref_pair_weight"] = torch.tensor(weight, dtype=torch.float32)
                row["agent_id"] = winner_id
                # preference rows serve ONLY the Bradley-Terry loss: zero economic credit,
                # zero reward; dp_actor drops them from PG/entropy/KL by role != 0
                row["rewards"] = 0.0
                row["econ_credit_internal"] = 0.0
                row["econ_is_final"] = 0.0
                row["econ_credit"] = 0.0
                row["active_masks"] = True
                row["agent_active_mask"] = True
                rows.append(row)
            out.append((s, rows, dict(driver.last_meta)))
            self.last_stats["pairs"] += 1
            if _rec is not None:
                self._dump({**_rec, "outcome": "pair", "pair_id": pid,
                            "y_w_text": pair.y_w_text, "y_l_text": pair.y_l_text,
                            "weight": weight, "meta": dict(driver.last_meta),
                            "traj_a": env_a.traj, "traj_b": env_b.traj})

        print(f"[cf-collector] iteration done: {self.last_stats}", flush=True)
        self._dump({"type": "iteration_stats", "stats": dict(self.last_stats)})
        return out

    def branch_winner(self, quotes):
        """Clear ONE branch continuation turn exactly as ``ChainTurnEngine.run_turn`` clears a
        mainline turn, and return the winning agent index.

        ``quotes[i]`` is agent i's parsed bid, or None when its answer carried no ``<bid>``
        tag -- which the mainline also treats as a pass. The mainline sequence is: snap every
        quote to the level grid, then (bid_pass) make 0.0 ineligible and reopen an all-pass
        turn at the floor for EVERY bidder, then break exact ties uniformly at random via
        ``rank_bids``'s jitter.

        The hand-written ``max`` this replaces let a pass win, never reopened an all-pass turn
        and handed every tie to agent index 0: in the first production preference iteration 79
        of 80 branch continuation turns went to the Logger agent, so both branch scores measured
        a Logger monopoly instead of the continuation the mainline policy would have produced
        (mainline trace: 7.0% of turns are top-bid ties and 5.4% are all-pass reopens, and both
        go to Miner/Smith about two thirds of the time).

        A method rather than a closure so the semantics are unit-testable without a live
        orchestra, env server or GPU.
        """
        orch = self.orch
        n = len(quotes)
        raw = [np.zeros(1, dtype=np.float64) for _ in range(n)]
        vals = np.full((n, 1), -np.inf, dtype=np.float64)
        bidders = [np.ones(1, dtype=bool) for _ in range(n)]
        active = np.ones(1, dtype=bool)
        for i, q in enumerate(quotes):
            raw[i][0] = 0.0 if q is None else float(q)
            vals[i][0] = raw[i][0]
        levels = list(getattr(orch, "bid_levels", None) or [])
        if levels:
            snap_bids_to_levels(raw, vals, bidders, active, levels)
            if getattr(orch, "bid_pass", False):
                apply_bid_pass(raw, vals, bidders, active, levels)
        return int(rank_bids(vals, self._branch_rng)[0])

    def _snap_level(self, bid):
        levels = list(self.orch.bid_levels or [])
        if not levels:
            return float(bid)
        return min((float(l) for l in levels), key=lambda lv: abs(lv - float(bid)))


# Older names, kept for imports: one collector serves every environment.
MinecraftCounterfactualCollector = CounterfactualCollector
NavilaCounterfactualCollector = CounterfactualCollector
