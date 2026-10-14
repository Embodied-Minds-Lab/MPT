import argparse
import base64
import copy
import importlib.util
import io
import json
import os
import re
import sys
import traceback

import numpy as np
from aiohttp import web
from PIL import Image

# Where the NaVILA eval tree lives (vlnce_baselines + habitat_extensions + the relative data/ dir):
# the `evaluation/` directory of a NaVILA checkout. Required.
EVAL_DIR = os.environ.get("NAVILA_EVAL_DIR", "")
if not EVAL_DIR:
    raise RuntimeError("NAVILA_EVAL_DIR is not set: point it at the `evaluation/` directory of a NaVILA "
                       "checkout (vlnce_baselines + habitat_extensions + its data/ dir).")
sys.path.insert(0, EVAL_DIR)

# vlnce_baselines/__init__.py imports navila_trainer, which imports llava (the VILA package).
# This server never runs the policy -- sglang does, in the trainer env -- so on installs without
# VILA we register the package WITHOUT executing its __init__ body; the two submodules below
# import fine on their own, and habitat_extensions (pulled in by config.default) still registers
# WaypointRewardMeasure. Set NAVILA_SKIP_VLNCE_INIT=0 to restore the plain import.
if os.environ.get("NAVILA_SKIP_VLNCE_INIT", "1") == "1" and "vlnce_baselines" not in sys.modules:
    _pkg_dir = os.path.join(EVAL_DIR, "vlnce_baselines")
    _spec = importlib.util.spec_from_file_location(
        "vlnce_baselines", os.path.join(_pkg_dir, "__init__.py"),
        submodule_search_locations=[_pkg_dir])
    sys.modules["vlnce_baselines"] = importlib.util.module_from_spec(_spec)  # body deliberately not run

from vlnce_baselines.config.default import get_config
from vlnce_baselines.common.environments import VLNCEWaypointEnv
from habitat.datasets import make_dataset

NUM_FRAMES = 8
CONFIG_PATH = os.environ.get(
    "NAVILA_CONFIG_PATH",
    os.path.join(EVAL_DIR, "vlnce_baselines/config/r2r_baselines/navila.yaml"))

# --- mid-level action parsing (ports navila_trainer.py) -> atomic habitat action ints ---
STOP, FORWARD, LEFT, RIGHT = 0, 1, 2, 3
_PATTERNS = {
    STOP:    re.compile(r"\bstop\b", re.IGNORECASE),
    FORWARD: re.compile(r"\bmove forward\b", re.IGNORECASE),
    LEFT:    re.compile(r"\bturn left\b", re.IGNORECASE),
    RIGHT:   re.compile(r"\bturn right\b", re.IGNORECASE),
}

# NAVILA_FAIL_STOP: extra reward added when the agent CHOOSES stop but has NOT reached the goal
# (won=False). Default 0.0 (off). Set NAVILA_FAIL_STOP=-0.05 to penalise premature/lazy stopping so
# that immediate-stop is worse than attempting to navigate -- counters the stop-collapse local
# optimum. Only fires on an actual stop action ([STOP]), never on a max-steps timeout.
_FAIL_STOP_PENALTY = float(os.environ.get("NAVILA_FAIL_STOP", 0.0))


# NAVILA_SNAPSHOT=1 enables the /snapshot + /restore endpoints (mid-episode branching).
# Default OFF: nothing running changes behavior; the endpoints answer {"error": "snapshot disabled"}.
_SNAPSHOT_ON = os.environ.get("NAVILA_SNAPSHOT", "0") == "1"

_ACT_TAG = re.compile(r"<act>(.*?)</act>", re.IGNORECASE | re.DOTALL)


def parse_action(text):
    """Return a list of atomic action ints (queue expansion: 45deg -> 3x15deg, 75cm -> 3x25cm),
    or None if the text names no known verb. The model emits <think>...</think>\\n<act>ACTION</act>;
    we parse the ACTION inside <act> (so a verb mentioned in <think> can't be mistaken for the action)."""
    t = text or ""
    m_act = _ACT_TAG.search(t)
    if m_act:
        t = m_act.group(1)          # only the action span
    verb = None
    for a, p in _PATTERNS.items():
        if p.search(t):
            verb = a
            break
    if verb is None:
        return None
    if verb == STOP:
        return [STOP]
    if verb == FORWARD:
        m = re.search(r"move forward (\d+)\s*cm", t, re.IGNORECASE)
        dist = int(m.group(1)) if m else 25
        dist = min([25, 50, 75], key=lambda x: abs(x - dist)) if dist % 25 else dist
        return [FORWARD] * max(1, dist // 25)
    # LEFT / RIGHT
    m = re.search(r"turn (?:left|right) (\d+)\s*degree", t, re.IGNORECASE)
    deg = int(m.group(1)) if m else 15
    deg = min([15, 30, 45], key=lambda x: abs(x - deg)) if deg % 15 else deg
    return [verb] * max(1, deg // 15)


def sample_and_pad(images, num_frames=NUM_FRAMES, width=512, height=512):
    """Ports navila_trainer.sample_and_pad_images: pad the front with black to >=num_frames, then
    take num_frames-1 linspace-sampled history frames + the latest."""
    frames = list(images)
    while len(frames) < num_frames:
        frames.insert(0, Image.new("RGB", (width, height), (0, 0, 0)))
    latest = frames[-1]
    idxs = np.linspace(0, len(frames) - 1, num=num_frames - 1, endpoint=False, dtype=int)
    return [frames[i] for i in idxs] + [latest]


def png_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def jpg_b64(img, quality=70):
    """History frames inside snapshots: JPEG q70 is ~10x smaller than PNG and the window re-encode
    after restore stays visually equivalent (the model consumes 512x512 renders, not pixel art)."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def _rgb(obs):
    return Image.fromarray(np.uint8(obs["rgb"])).convert("RGB")


def _b64_rgb(b):
    return Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB")


# --- snapshot/restore state (de)serialization -------------------------------------------------
# Per-measure attributes that are static (sim/config handles, GT tables) or fully rebuilt by
# reset_metric on the same episode -- never worth shipping in a snapshot. Everything else in
# vars(measure) is captured generically if _enc_state can encode it (covers the episode-local
# state of the stock VLN-CE measures: DistanceToGoal/_previous_position, SPL/_agent_episode_
# distance, NDTW/locations, PathLength, StepsTaken, Oracle*, and WaypointRewardMeasure's
# _previous_distance_to_goal/_prev_position/_metric).
_MEASURE_SKIP_ATTRS = {"_sim", "_config", "uuid", "gt_json", "dtw_func", "_conn_graphs",
                       "_episode_view_points", "gt_locations"}

# WaypointRewardMeasure's config-derived reward scalars. They travel WITH the snapshot on purpose:
# a branch must replay under the reward config it was snapshotted under (otherwise probe-reward
# equality breaks across servers with different NAVILA_SLACK), so restore overwrites the target's
# values for the rest of that episode -- loudly, whenever they differ (see Slot.restore).
_REWARD_CFG_ATTRS = ("_slack_reward", "_success_reward", "_distance_scalar",
                     "_use_distance_scaled_slack_reward", "_scale_slack_on_prediction")
_MEASURE_SKIP_ATTRS |= set(_REWARD_CFG_ATTRS)  # reward cfg travels via the explicit snap["reward_cfg"]
# path below (with its override warning); keep the generic measure walk away from these five scalars
# so that path is the single writer and the warning can actually fire.
_WAYPOINT_UUID = "waypoint_reward_measure"

# once-per-key logs for the degradation paths (dropped attr at snapshot, unknown uuid at restore,
# imported reward config) -- a silent skip would produce plausible-looking but state-divergent
# branches with no trace.
_WARNED_KEYS = set()


def _warn_once(key, msg):
    if key not in _WARNED_KEYS:
        _WARNED_KEYS.add(key)
        print(f"[navila-srv] {msg}", flush=True)


def _enc_state(v):
    """JSON-encode one piece of measure state; raise TypeError on non-episode-local values
    (functions, sim handles, ...) so the caller skips that attribute."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (np.bool_, np.integer, np.floating)):
        return v.item()
    if isinstance(v, np.ndarray):
        return {"__nd__": base64.b64encode(np.ascontiguousarray(v).tobytes()).decode(),
                "dtype": str(v.dtype), "shape": list(v.shape)}
    if type(v).__name__ == "quaternion":  # np-quaternion (agent rotations)
        return {"__quat__": [float(v.x), float(v.y), float(v.z), float(v.w)]}
    if isinstance(v, tuple):
        return {"__tuple__": [_enc_state(x) for x in v]}
    if isinstance(v, list):
        return [_enc_state(x) for x in v]
    if isinstance(v, dict) and all(isinstance(k, str) for k in v):
        return {k: _enc_state(x) for k, x in v.items()}
    raise TypeError(f"unsnapshotable type {type(v)}")


def _dec_state(v):
    """Inverse of _enc_state."""
    if isinstance(v, list):
        return [_dec_state(x) for x in v]
    if isinstance(v, dict):
        if "__nd__" in v:
            a = np.frombuffer(base64.b64decode(v["__nd__"]), dtype=np.dtype(v["dtype"]))
            return a.reshape(v["shape"]).copy()
        if "__quat__" in v:
            import quaternion
            x, y, z, w = v["__quat__"]
            return quaternion.quaternion(w, x, y, z)  # np-quaternion ctor is (w, x, y, z)
        if "__tuple__" in v:
            return tuple(_dec_state(x) for x in v["__tuple__"])
        return {k: _dec_state(x) for k, x in v.items()}
    return v


class WaypointEnv(VLNCEWaypointEnv):
    """VLNCEWaypointEnv but get_reward_range avoids the removed np.float (numpy>=1.24). We keep the
    official get_reward/get_done/get_info unchanged -- only the range accessor is patched for the
    newer numpy in this env."""

    def get_reward_range(self):
        return (float("-inf"), float("inf"))


class Slot:
    """One habitat env running one R2R-CE episode, with the NaVILA frame-window + queue state."""

    def __init__(self, config, dataset):
        self.env = WaypointEnv(config, dataset)
        self.ep_by_id = {str(ep.episode_id): ep for ep in self.env.episodes}
        self.past_rgbs = []
        self.instruction = ""
        self.n_atomic = 0
        self.max_steps = int(config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS)
        self.done = True

    def _window(self, curr):
        return [png_b64(im) for im in sample_and_pad(self.past_rgbs + [curr])]

    def reset(self, kw):
        ep_id = kw.get("episode_id")
        if ep_id is not None and str(ep_id) in self.ep_by_id:
            self.env._env.episode_iterator = iter([self.ep_by_id[str(ep_id)]])
        try:
            obs = self.env.reset()
        except StopIteration:
            # An id-addressed reset (or a restore) leaves a single-episode iterator behind,
            # exhausted; the next id-LESS reset (the {} padding of a spare slot in the /reset
            # fanout) then pulls from it and dies here. Re-seed with any known episode and
            # retry once -- which episode is irrelevant, the slot is only parked.
            first = next(iter(self.ep_by_id.values()))
            self.env._env.episode_iterator = iter([first])
            obs = self.env.reset()
        ep = self.env._env.current_episode
        self.instruction = ep.instruction.instruction_text
        self.past_rgbs = []
        self.n_atomic = 0
        self.done = False
        # terminal info cache for the post-done short-circuit in step(): without it, a branch
        # that SUCCEEDS before its budget runs out reports won:False on every subsequent
        # padded step, and a driver that reads the final observation scores the success as a
        # failure.
        self._final_info = None
        self._curr = _rgb(obs)
        return {"text": self.instruction, "frames_b64": self._window(self._curr),
                "info": {"data_source": "navila", "episode_id": str(ep.episode_id)}}

    def step(self, action_text):
        if self.done:
            # Post-done short-circuit. The terminal step's info is replayed VERBATIM (won and
            # all measures) instead of a hardcoded won:False -- an episode that already
            # succeeded must keep saying so, or an early-stopping branch reads as a failure
            # to whoever inspects the final observation. Reward stays 0.0: nothing new
            # happened; only is_action_valid is forced off.
            info = dict(getattr(self, "_final_info", None)
                        or {"data_source": "navila", "won": False})
            info["is_action_valid"] = False
            return {"text": "", "reward": 0.0, "done": True,
                    "frames_b64": self._window(self._curr), "info": info}
        actions = parse_action(action_text)
        if actions is None:  # malformed -> no move, run down the step budget
            self.n_atomic += 1
            done = self.n_atomic >= self.max_steps
            self.done = done
            info = {"data_source": "navila", "won": False, "is_action_valid": False,
                    **self._measures()}
            if done:
                self._final_info = dict(info)
            return {"text": self.instruction, "reward": 0.0, "done": done,
                    "frames_b64": self._window(self._curr), "info": info}
        reward_sum = 0.0
        done = False
        try:
            for a in actions:
                # append the PRE-step observation (navila_trainer order), then step
                self.past_rgbs.append(self._curr)
                obs, r, d, info = self.env.step(a)
                reward_sum += float(r)
                self.n_atomic += 1
                self._curr = _rgb(obs)
                if d or self.n_atomic >= self.max_steps:
                    done = True
                    break
        except Exception as e:
            self.done = True
            info = {"data_source": "navila", "won": False, "error": str(e)}
            self._final_info = dict(info)
            return {"text": f"STEP ERROR: {e}", "reward": 0.0, "done": True,
                    "frames_b64": self._window(self._curr), "info": info}
        if not np.isfinite(reward_sum):
            reward_sum, done = 0.0, True
        self.done = done
        won = bool(self.env._episode_success())
        # failure-stop penalty: the agent chose stop but did not reach the goal
        if _FAIL_STOP_PENALTY and actions == [STOP] and not won:
            reward_sum += _FAIL_STOP_PENALTY
        info = {"data_source": "navila", "won": won, "is_action_valid": True,
                **self._measures()}
        if done:
            self._final_info = dict(info)
        return {"text": self.instruction, "reward": reward_sum, "done": done,
                "frames_b64": self._window(self._curr), "info": info}

    def snapshot(self):
        """Clone-on-demand branch point: a JSON-serializable dict of EVERYTHING episode-local.
        Restorable onto any slot of a server on the same split (all slots share ep_by_id)."""
        env = self.env._env
        ep = env.current_episode
        st = env.sim.get_agent_state()
        measures = {}
        dropped = []
        for uuid, m in env.task.measurements.measures.items():
            mstate = {}
            for k, v in vars(m).items():
                if k in _MEASURE_SKIP_ATTRS:
                    continue
                try:
                    mstate[k] = _enc_state(v)
                except TypeError:
                    # sim/config handles etc. -- static, rebuilt by reset. Recorded + logged: a
                    # future stateful-but-unencodable attribute must not vanish silently.
                    dropped.append(f"{uuid}.{k}")
                    _warn_once(("drop", uuid, k),
                               f"snapshot: dropping non-encodable measure attr {uuid}.{k} "
                               f"({type(v).__name__}) -- restore relies on reset for it")
            measures[uuid] = mstate
        # reward config, captured explicitly (see _REWARD_CFG_ATTRS): restore compares + logs it
        wp = env.task.measurements.measures.get(_WAYPOINT_UUID)
        reward_cfg = {} if wp is None else {k: _enc_state(getattr(wp, k))
                                            for k in _REWARD_CFG_ATTRS if hasattr(wp, k)}
        snap = {
            "episode_id": str(ep.episode_id),
            "agent": {"position": [float(x) for x in st.position],
                      "rotation": [float(st.rotation.x), float(st.rotation.y),
                                   float(st.rotation.z), float(st.rotation.w)]},
            "elapsed_steps": int(env._elapsed_steps),
            "episode_over": bool(env._episode_over),
            "is_stop_called": bool(getattr(env.task, "is_stop_called", False)),
            "is_episode_active": bool(getattr(env.task, "_is_episode_active", True)),
            "n_atomic": int(self.n_atomic),
            "done": bool(self.done),
            "instruction": self.instruction,
            # the FULL atomic history must ship: the 8-frame window linspace-samples the WHOLE
            # list, so any frame can be referenced by a later window. JPEG q70 keeps that
            # affordable. No current frame -- restore re-renders it at the restored pose.
            "past_rgbs_b64": [jpg_b64(im) for im in self.past_rgbs],
            "reward_cfg": reward_cfg,
            "dropped_attrs": dropped,   # [] on the stock measure set; client-side assertable
            "measures": measures,
        }
        return snap

    def restore(self, snap):
        """Two phases: (1) rebuild the episode context via the normal reset path + teleport the
        agent + overwrite measure/counter internals from the snapshot; (2) read: render the
        observation at the restored pose and return a step-shaped payload for the driver."""
        # phase 1a: reset to the SAME episode (rebuilds scene, task, measures, GT tables).
        # Guard: Slot.reset silently falls through to the iterator on an unknown id -- for a
        # restore that would corrupt the branch, so fail loudly (snapshots only travel between
        # slots serving the same split).
        if str(snap["episode_id"]) not in self.ep_by_id:
            raise KeyError(f"episode_id {snap['episode_id']} not in this slot's split")
        self.reset({"episode_id": snap["episode_id"]})
        env = self.env._env
        # phase 1b: teleport; keep_agent_at_new_pose leaves the sim in the restored state, and the
        # returned obs is rendered AT that pose (rotation as [x, y, z, w], habitat-sim converts)
        obs = env.sim.get_observations_at(position=snap["agent"]["position"],
                                          rotation=snap["agent"]["rotation"],
                                          keep_agent_at_new_pose=True)
        # phase 1c: habitat Env / task counters
        env._elapsed_steps = int(snap["elapsed_steps"])
        env._episode_over = bool(snap["episode_over"])
        if hasattr(env.task, "is_stop_called"):
            env.task.is_stop_called = bool(snap.get("is_stop_called", False))
        env.task._is_episode_active = bool(snap.get("is_episode_active", True))
        # phase 1d: measure internals (WaypointRewardMeasure._previous_distance_to_goal,
        # DistanceToGoal._previous_position, SPL/NDTW/PathLength cumulatives, ...)
        for uuid, mstate in snap.get("measures", {}).items():
            m = env.task.measurements.measures.get(uuid)
            if m is None:
                # measure-set mismatch between source and target server: a silent skip would give
                # a plausible-looking but state-divergent branch, so say it (once per uuid)
                _warn_once(("miss", uuid),
                           f"restore: snapshot carries measure {uuid!r} this server does not "
                           f"run -- its state is NOT restored")
                continue
            for k, v in mstate.items():
                setattr(m, k, _dec_state(v))
        # reward config travels with the snapshot by design (see _REWARD_CFG_ATTRS): overwrite the
        # target's scalars, loudly whenever they differ from what this server was configured with.
        wp = env.task.measurements.measures.get(_WAYPOINT_UUID)
        if wp is not None:
            for k, v in (snap.get("reward_cfg") or {}).items():
                v = _dec_state(v)
                old = getattr(wp, k, None)
                if old is not None and old != v:
                    _warn_once(("cfg", k, repr(v)),
                               f"restore: snapshot reward config {k}={v} overrides this "
                               f"server's configured {old} for the rest of the episode")
                setattr(wp, k, v)
        # phase 1e: slot-side counters + frame history
        self.instruction = snap.get("instruction", self.instruction)
        self.n_atomic = int(snap["n_atomic"])
        self.done = bool(snap["done"])
        # the terminal-info cache belongs to the PREVIOUS timeline; a branch resumes
        # mid-episode (snapshotting a done state is a caller error the driver's scoring
        # would surface as a permanent tie anyway)
        self._final_info = None
        self.past_rgbs = [_b64_rgb(b) for b in snap.get("past_rgbs_b64", [])]
        self._curr = _rgb(obs)
        # phase 2: step-shaped payload so the driver can continue stepping immediately
        return {"text": self.instruction, "reward": 0.0, "done": self.done,
                "frames_b64": self._window(self._curr),
                "info": {"data_source": "navila", "episode_id": snap["episode_id"],
                         "restored": True, "is_action_valid": True, **self._measures()}}

    def _measures(self):
        m = self.env.habitat_env.get_metrics()
        out = {}
        for k in ("spl", "ndtw", "success", "distance_to_goal", "oracle_success", "path_length"):
            if k in m and m[k] is not None:
                try:
                    out[k] = float(m[k])
                except (TypeError, ValueError):
                    pass
        return out


def build_config(split, sim_gpu):
    config = get_config(CONFIG_PATH, [])
    config.defrost()
    config.SIMULATOR_GPU_IDS = [sim_gpu]
    config.TORCH_GPU_ID = sim_gpu
    tc = config.TASK_CONFIG
    # DATASET fields the NaVILA loader (habitat_extensions/task.py) requires, set exactly as
    # navila_trainer does (no chunking; single guide instruction).
    tc.DATASET.SPLIT = split
    tc.DATASET.ROLES = ["guide"]
    tc.DATASET.LANGUAGES = list(config.EVAL.LANGUAGES)
    tc.DATASET.NUM_CHUNKS = 1
    tc.DATASET.CHUNK_IDX = 0
    # add the official RL reward measure to the discrete-action task
    if "WAYPOINT_REWARD_MEASURE" not in tc.TASK.MEASUREMENTS:
        tc.TASK.MEASUREMENTS = list(tc.TASK.MEASUREMENTS) + ["WAYPOINT_REWARD_MEASURE"]
    # RL block consumed by VLNCEWaypointEnv.get_reward / _episode_success
    config.RL.REWARD_MEASURE = "waypoint_reward_measure"
    config.RL.SUCCESS_MEASURE = "success"
    # experimental knob: override the per-step slack penalty. NAVILA_SLACK=0 removes the official
    # -0.05 that makes early-stop the reward-minimising local optimum (the collapse we hit); unset
    # keeps the official -0.05. Progress reward stays potential-based, so this cannot be oscillation-
    # hacked.
    tc.TASK.WAYPOINT_REWARD_MEASURE.slack_reward = float(
        os.environ.get("NAVILA_SLACK", tc.TASK.WAYPOINT_REWARD_MEASURE.slack_reward))
    config.freeze()
    return config


def _worker(idx, split, sim_gpu, conn):
    """One slot = one habitat sim in its OWN process (habitat-sim allows only one GL context per
    process, so N sims must be N processes). Builds config+dataset+Slot, then serves reset/step
    over the pipe until told to stop."""
    try:
        config = build_config(split, sim_gpu)
        dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET)
        slot = Slot(config, dataset)
        conn.send(("ready", idx))
    except Exception as e:
        conn.send(("error", f"slot {idx} build failed: {e}\n{traceback.format_exc()}"))
        return
    while True:
        try:
            cmd, arg = conn.recv()
        except EOFError:
            break
        if cmd == "stop":
            break
        try:
            if cmd == "snapshot":       # arg None = not this slot's turn in the fanout -> no-op
                out = None if arg is None else slot.snapshot()
            elif cmd == "restore":
                out = None if arg is None else slot.restore(arg)
            elif cmd == "step_slot":    # slot-addressed step: non-target slots no-op on None
                out = None if arg is None else slot.step(arg)
            elif cmd == "reset":
                out = slot.reset(arg)
            else:
                # plain "step": a None arg means this slot is branch-owned and sits the
                # mainline fanout out. Without this branch, parse_action(None) raised inside
                # step(), the except path answered done:True, and the mainline would appear
                # to have terminated the branch slot it was supposed to leave alone.
                out = None if arg is None else slot.step(arg)
        except Exception as e:
            traceback.print_exc()
            if cmd in ("snapshot", "restore"):
                out = {"error": f"{cmd} failed: {e}"}
            else:
                out = {"text": f"SLOT ERROR: {e}", "reward": 0.0, "done": True,
                       "frames_b64": [], "info": {"data_source": "navila", "won": False}}
        conn.send(("ok", out))


def make_app(n_slots, split, num_gpus):
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    conns, procs = [], []
    print(f"[navila-srv] spawning {n_slots} slot workers over {num_gpus} GPU(s) (split={split})...",
          flush=True)
    for i in range(n_slots):
        sim_gpu = i % max(1, num_gpus)   # spread habitat slots across the render GPUs
        parent, child = ctx.Pipe()
        p = ctx.Process(target=_worker, args=(i, split, sim_gpu, child), daemon=True)
        p.start()
        conns.append(parent)
        procs.append(p)
    # wait for all workers to finish building their sim
    for i, c in enumerate(conns):
        tag, payload = c.recv()
        if tag == "error":
            raise RuntimeError(payload)
        print(f"[navila-srv] slot {i} ready", flush=True)

    def _fanout(cmd, args):
        for c, a in zip(conns, args):
            c.send((cmd, a))
        return [c.recv()[1] for c in conns]

    async def health(_req):
        # navila_snapshot: whether /snapshot /restore /step_slot are enabled (the counterfactual
        # bid phase of training needs "1"; the trainer's env client checks it at startup)
        return web.json_response({"ok": True, "slots": len(conns),
                                  "navila_snapshot": os.environ.get("NAVILA_SNAPSHOT", "0")})

    import asyncio
    import collections as _collections

    # ONE lock for ALL slot-worker pipe traffic: _fanout and _one_slot send/recv on EVERY conn
    # (the lockstep discipline), so per-slot locks cannot serialize the shared pipe set -- an
    # overlapping /step fanout and a /snapshot|/restore|/step_slot call would otherwise interleave
    # send/recv on the same Pipes and cross-wire replies for all slots.
    _pipe_lock = asyncio.Lock()

    # Slots currently owned by a counterfactual branch. Set by /restore, cleared by /reset
    # (every /reset fanout re-purposes all slots) and by a later /restore of the same slot.
    # The mainline /step fanout pads "" into every slot beyond the request -- for an ordinary
    # spare slot that is harmless budget burn on a parked episode, but for a slot mid-branch
    # each mainline step would advance (and eventually terminate) the branch behind the
    # driver's back. Branch-owned slots are therefore excluded from the fanout entirely
    # (worker no-op via the None argument, same discipline as _one_slot).
    _branch_owned = [False] * len(conns)

    # NAVILA_SUCCESS_DUMP=<dir>: buffer every slot's (action, reward, current frame) and, when an
    # episode ends in success (won=True), write its frames + meta.json to <dir>/ep_XXXX for a
    # rendered per-step visualisation. Only successful episodes are kept, so this is cheap.
    _succ_dump = os.environ.get("NAVILA_SUCCESS_DUMP", "")
    _ep_bufs = _collections.defaultdict(list)
    _succ_n = [0]

    def _cur_frame(o):
        fb = o.get("frames_b64") or []
        return fb[-1] if fb else None

    def _dump_success(buf):
        n = _succ_n[0]; _succ_n[0] += 1
        d = os.path.join(_succ_dump, f"ep_{n:04d}")
        os.makedirs(d, exist_ok=True)
        meta = []
        for i, s in enumerate(buf):
            if s.get("frame"):
                try:
                    open(os.path.join(d, f"{i:03d}.png"), "wb").write(base64.b64decode(s["frame"]))
                except Exception:
                    pass
            meta.append({k: s.get(k) for k in ("action", "reward", "won", "instruction")})
        json.dump(meta, open(os.path.join(d, "meta.json"), "w"), ensure_ascii=False)

    async def reset(req):
        body = await req.json()
        n = len(body.get("slots", []))
        kws = list(body.get("slots", []))[:len(conns)]
        kws += [{}] * (len(conns) - len(kws))
        for i in range(len(_branch_owned)):
            _branch_owned[i] = False          # a reset fanout re-purposes every slot
        async with _pipe_lock:
            obs = await asyncio.get_running_loop().run_in_executor(None, _fanout, "reset", kws)
        if _succ_dump:
            for i, o in enumerate(obs):
                _ep_bufs[i] = [{"action": "START", "instruction": o.get("text", ""),
                                "frame": _cur_frame(o)}]
        return web.json_response({"obs": obs[:n]})

    _act_trace = os.environ.get("NAVILA_ACT_TRACE", "")

    async def step(req):
        body = await req.json()
        n = len(body.get("actions", []))
        acts = list(body.get("actions", []))[:len(conns)]
        acts += [""] * (len(conns) - len(acts))
        # branch-owned slots sit the fanout out (None -> worker no-op). A mainline that
        # explicitly addresses a branch-owned slot (index < n) gets None back in its obs
        # list -- that is a client-side layout bug and should crash there, not silently
        # step someone else's branch.
        acts = [None if _branch_owned[i] else a for i, a in enumerate(acts)]
        async with _pipe_lock:
            obs = await asyncio.get_running_loop().run_in_executor(None, _fanout, "step", acts)
        if _act_trace and obs:  # capture slot-0's model output (think+act) + reward for I/O viz
            try:
                o0 = obs[0]; i0 = o0.get("info", {}) or {}
                with open(_act_trace, "a") as _f:
                    _f.write(json.dumps({"action": str(acts[0])[:400], "reward": o0.get("reward"),
                                         "done": o0.get("done"), "won": i0.get("won"),
                                         "valid": i0.get("is_action_valid"),
                                         "episode_id": i0.get("episode_id")}) + "\n")
            except Exception:
                pass
        if _succ_dump and obs:  # buffer per-slot; dump frames+actions when an episode succeeds
            for i, o in enumerate(obs):
                if o is not None:             # branch-owned slots no-op the fanout as None
                    _buf_step(i, acts[i], o)
        return web.json_response({"obs": obs[:n]})

    def _buf_step(i, action, o):
        """Success-dump bookkeeping for one slot's step result (shared by /step and /step_slot)."""
        inf = o.get("info", {}) or {}
        _ep_bufs[i].append({"action": str(action)[:400], "reward": o.get("reward"),
                            "won": inf.get("won"), "frame": _cur_frame(o)})
        if o.get("done"):
            if inf.get("won") and _ep_bufs[i]:
                try:
                    _dump_success(_ep_bufs[i])
                except Exception:
                    pass
            _ep_bufs[i] = []

    def _one_slot(cmd, i, arg):
        """Run cmd on slot i only, through the same fanout discipline (every conn gets exactly one
        send+recv per call, non-target slots no-op on None) -- keeps the pipes in lockstep."""
        args = [None] * len(conns)
        args[i] = arg
        return _fanout(cmd, args)[i]

    async def snapshot(req):
        if not _SNAPSHOT_ON:
            return web.json_response({"error": "snapshot disabled"})
        body = await req.json()
        i = int(body.get("slot", -1))
        if not 0 <= i < len(conns):
            return web.json_response({"error": f"bad slot {i}"})
        async with _pipe_lock:
            out = await asyncio.get_running_loop().run_in_executor(
                None, _one_slot, "snapshot", i, {})
        if isinstance(out, dict) and "error" in out:
            return web.json_response(out)
        # snapshot bodies carry the full frame history (MBs): dump off the event loop so /health
        # and concurrent requests do not stall on the serialization
        text = await asyncio.get_running_loop().run_in_executor(None, json.dumps, {"state": out})
        return web.Response(text=text, content_type="application/json")

    async def restore(req):
        if not _SNAPSHOT_ON:
            return web.json_response({"error": "snapshot disabled"})
        raw = await req.read()  # parse the (large) snapshot body off the event loop too
        body = await asyncio.get_running_loop().run_in_executor(None, json.loads, raw)
        i = int(body.get("slot", -1))
        state = body.get("state")
        if not 0 <= i < len(conns) or not isinstance(state, dict):
            return web.json_response({"error": "need slot index + state dict"})
        async with _pipe_lock:
            out = await asyncio.get_running_loop().run_in_executor(
                None, _one_slot, "restore", i, state)
        if isinstance(out, dict) and "error" in out:
            return web.json_response(out)
        # the slot now belongs to a branch: exclude it from the mainline /step fanout until
        # the next /reset re-purposes it (see _branch_owned above)
        _branch_owned[i] = True
        if _succ_dump and isinstance(out, dict):  # a restore rewinds the episode -> restart the buffer
            _ep_bufs[i] = [{"action": "RESTORE", "instruction": out.get("text", ""),
                            "frame": _cur_frame(out)}]
        return web.json_response({"obs": out})

    async def step_slot(req):
        """Slot-addressed step for driving a restored branch: exactly ONE slot steps and returns
        its step-shaped payload; every other slot's worker no-ops (None), so branch driving never
        pads "" into the mainline slots and burns their step budget."""
        if not _SNAPSHOT_ON:
            return web.json_response({"error": "snapshot disabled"})
        body = await req.json()
        i = int(body.get("slot", -1))
        if not 0 <= i < len(conns):
            return web.json_response({"error": f"bad slot {i}"})
        action = str(body.get("action", ""))
        async with _pipe_lock:
            out = await asyncio.get_running_loop().run_in_executor(
                None, _one_slot, "step_slot", i, action)
        if _succ_dump and isinstance(out, dict):
            _buf_step(i, action, out)
        return web.json_response({"obs": out})

    app = web.Application(client_max_size=256 * 1024 * 1024)
    app.add_routes([web.get("/health", health), web.post("/reset", reset), web.post("/step", step),
                    web.post("/snapshot", snapshot), web.post("/restore", restore),
                    web.post("/step_slot", step_slot)])
    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=31600)
    ap.add_argument("--slots", type=int, default=1)
    ap.add_argument("--split", default="train")
    ap.add_argument("--num-gpus", type=int, default=1, help="render GPUs to spread slots across")
    args = ap.parse_args()
    app = make_app(args.slots, args.split, args.num_gpus)
    print(f"[navila-srv] serving on :{args.port} ({args.slots} slots, split={args.split})", flush=True)
    web.run_app(app, host="0.0.0.0", port=args.port)
