import argparse, asyncio, base64, contextlib, io, json, os, re, threading, time, traceback, uuid
import numpy as np
from aiohttp import web

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_ACTION = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)


# ---- render size (env-tunable). Training keeps the 640x360 default; the val server sets
# MC_RENDER=1280x720 so recorded video is genuinely high-res. Policy input (obs_size) unchanged.
_R = os.environ.get("MC_RENDER", "640x360").lower().split("x")
RENDER_SIZE = (int(_R[0]), int(_R[1]))

# ---- scripted semantic crafting (opt-in: MC_SCRIPTED_CRAFT=1) --------------------------------
# Why this wall exists: STEVE-1's own paper reports ~0% success on text-prompted GUI crafting
# (only visual prompt-chaining reaches planks), and across 2653 episodes of our runs GUI crafting
# produced exactly 1 wooden pickaxe and 0 stone events. So locomotion/mining stay with STEVE-1,
# while instructions whose FIRST WORD is "craft"/"equip" execute the recipe semantically via
# Minecraft commands (/clear ingredients + /give products), one high-level turn each, verified
# against the real item inventory. Recipes keep game-faithful quantities; 3x3 recipes require a
# crafting_table IN INVENTORY (not placed -- that is the labeled simplification). Default OFF:
# byte-identical behavior for every existing caller.
SCRIPTED = os.environ.get("MC_SCRIPTED_CRAFT", "0") == "1"
_RECIPES = {
    # consume: (pattern, count); "#log"/"#planks" match any wood variant. needs: present, not consumed.
    "planks":         {"consume": [("#log", 1)],                      "give": [("oak_planks", 4)],     "needs": []},
    "crafting_table": {"consume": [("#planks", 4)],                   "give": [("crafting_table", 1)], "needs": []},
    "stick":          {"consume": [("#planks", 2)],                   "give": [("stick", 4)],          "needs": []},
    "wooden_pickaxe": {"consume": [("#planks", 3), ("stick", 2)],     "give": [("wooden_pickaxe", 1)], "needs": ["crafting_table"]},
    "stone_pickaxe":  {"consume": [("cobblestone", 3), ("stick", 2)], "give": [("stone_pickaxe", 1)], "needs": ["crafting_table"]},
    # ---- extended ladder toward diamond pickaxe (same labeled wall) ----
    "furnace":        {"consume": [("cobblestone", 8)],                  "give": [("furnace", 1)],         "needs": []},
    # smelting wall: furnace in inventory + planks as fuel; 1 ore -> 1 ingot
    "iron_ingot":     {"consume": [("iron_ore", 1), ("#fuel", 1)],       "give": [("iron_ingot", 1)],      "needs": ["furnace"]},
    "iron_pickaxe":   {"consume": [("iron_ingot", 3), ("stick", 2)],     "give": [("iron_pickaxe", 1)],    "needs": ["crafting_table"]},
    "diamond_pickaxe":{"consume": [("diamond", 3), ("stick", 2)],        "give": [("diamond_pickaxe", 1)], "needs": ["crafting_table"]},
    # reuse-task branches (real MC recipes, additive; unreachable unless the instruction
    # names them, so existing tasks are untouched):
    # torchcraft: smelt a log into charcoal (fuel separate), torch = charcoal + stick -> 4
    "charcoal":       {"consume": [("#log", 1), ("#fuel", 1)],           "give": [("charcoal", 1)],        "needs": ["furnace"]},
    "torch":          {"consume": [("charcoal", 1), ("stick", 1)],       "give": [("torch", 4)],           "needs": []},
    # stonebricks: smelt cobblestone back into stone, four stone -> four stone bricks
    "stone":          {"consume": [("cobblestone", 1), ("#fuel", 1)],    "give": [("stone", 1)],           "needs": ["furnace"]},
    "stone_bricks":   {"consume": [("stone", 4)],                        "give": [("stone_bricks", 4)],    "needs": ["crafting_table"]},
}

# 300, and it MUST stay well under every client's own timeout. A deadline that fires no earlier
# than the caller's is useless: the transfer client's /step timeout is 900s and this was also
# 900s, so the client gave up first, retried, and stacked a second hung request on the server
# while the first was still running -- three deadline firings and still zero steps served.
# Healthy numbers for reference: a step is 3-5s, a reset 12-92s. 300s is ~3x the worst observed.
RESET_DEADLINE_S = float(os.environ.get("MC_RESET_DEADLINE_S", "300"))


def _say(msg):
    """print() that can never fail a request.

    The server's stdout is often redirected to a file on a network filesystem. When that
    filesystem goes over quota the write-back fails and the kernel raises the deferred error on
    the next flush -- inside the /step handler, outside every try/except, so aiohttp turns it
    into a plain-text 500 and the trainer's bare r.json() dies on it, although every slot is
    healthy. Logging must never be load-bearing.
    """
    try:
        print(msg, flush=True)
    except OSError:
        pass


# 300, and it MUST stay well under every client's own timeout. A deadline that fires no earlier
# than the caller's is useless: the transfer client's /step timeout is 900s and this was also
# 900s, so the client gave up first, retried, and stacked a second hung request on the server
# while the first was still running -- three deadline firings and still zero steps served.
# Healthy numbers for reference: a step is 3-5s, a reset 12-92s. 300s is ~3x the worst observed.
RESET_DEADLINE_S = float(os.environ.get("MC_RESET_DEADLINE_S", "300"))


def _match_scripted(instr):
    """(verb, recipe|item) for 'craft X'/'equip X' first-word instructions, else None -> STEVE-1."""
    w = (instr or "").lower().split()
    if not w or w[0] not in ("craft", "equip", "smelt", "scatter"):
        return None
    s = " ".join(w)
    if w[0] == "scatter":
        return ("scatter", None)
    if w[0] == "equip":
        for it in ("diamond_pickaxe", "iron_pickaxe", "stone_pickaxe", "wooden_pickaxe", "crafting_table"):
            if it.replace("_", " ") in s or it in s:
                return ("equip", it)
        if "pickaxe" in s:
            return ("equip", "pickaxe")   # resolved against inventory at execution time
        return ("equip", None)
    if w[0] == "smelt":
        if "iron" in s: return ("craft", "iron_ingot")
        if "charcoal" in s or "log" in s: return ("craft", "charcoal")
        if "stone" in s or "cobble" in s: return ("craft", "stone")
        return ("craft", None)
    if "brick" in s:                    return ("craft", "stone_bricks")
    if "torch" in s:                    return ("craft", "torch")
    if "charcoal" in s:                 return ("craft", "charcoal")
    if "diamond" in s and "pickaxe" in s: return ("craft", "diamond_pickaxe")
    if "iron" in s and "pickaxe" in s:  return ("craft", "iron_pickaxe")
    if "stone" in s and "pickaxe" in s: return ("craft", "stone_pickaxe")
    if "pickaxe" in s:                  return ("craft", "wooden_pickaxe")
    if "furnace" in s:                  return ("craft", "furnace")
    if "table" in s:                    return ("craft", "crafting_table")
    if "stick" in s:                    return ("craft", "stick")
    if "plank" in s:                    return ("craft", "planks")
    return ("craft", None)

# ---- observation constants (apply to every caller) ---------------------------------------------
# The observation text carries a constant RULES block (mechanism facts and recipes), a
# Craftable-now line (mc_recipes.craftable_line), a truthful Last line with a failure hint, a
# no-progress WARNING and an output-contract line -- and NO reward number anywhere in the text.
# The format-SFT agents were trained on exactly this rendering: editing these constants changes
# the prompts the released checkpoints expect.
from mc_recipes import craftable_line as _craftable_line, ladder_for as _ladder_for, LADDER as _LADDER_IRON

# Wording verified line-by-line against _RECIPES: the furnace recipe has needs=[] (no crafting
# table required), and smelting needs the furnace IN THE INVENTORY with planks-or-coal fuel (#fuel
# matches no logs), reached by a "smelt ..." instruction -- this server's scripted protocol, not
# vanilla Minecraft's "place the furnace and load it".
# The shared wood-and-stone prefix. Identical for every chain, because every chain walks it.
RULES_SHARED = (
    "RULES (Minecraft facts, always true):\n"
    "- 1 log -> 4 planks (craft in your 2x2 grid, no table). 4 planks -> crafting table. 2 planks -> 4 sticks.\n"
    "- Wooden pickaxe = 3 planks + 2 sticks, at a crafting table.\n"
    "- Stone/cobblestone drops ONLY while a pickaxe is EQUIPPED (bare hands get nothing).\n"
    "- Stone pickaxe = 3 cobblestone + 2 sticks, at the table. Furnace = 8 cobblestone (no table needed).")

# The tail, selected by the task's OWN reward ladder, so a task is only told about the items of its
# own goal. Keying off reward_cfg rather than data_source is deliberate: the rollout drivers
# forward the ladder but NOT data_source, so a data_source key would silently fall back to iron
# on the eval path.
RULES_TAIL_IRON = (
    "\n- Iron ore drops ONLY while a STONE (or better) pickaxe is EQUIPPED; a wooden pickaxe breaks it but drops NOTHING.\n"
    "- Iron ore is found underground, mostly y=10..40 (you see your y in World).\n"
    "- Smelting: with the furnace and fuel (planks or coal) in your inventory, say \"smelt iron\" -> 1 iron ingot per ore.\n"
    "- Iron pickaxe = 3 iron ingots + 2 sticks, at the table.")
RULES_TAIL_TORCH = (
    "\n- Smelting: with the furnace and fuel (planks or coal) in your inventory, say \"smelt charcoal\" -> 1 charcoal per log burned.\n"
    "- Torch = 1 charcoal + 1 stick -> 4 torches (craft in your 2x2 grid, no table needed).\n"
    "- Keep spare logs: every charcoal costs one log, and there is no torch without charcoal.")
RULES_TAIL_BRICKS = (
    "\n- Smelting: with the furnace and fuel (planks or coal) in your inventory, say \"smelt stone\" -> 1 stone per cobblestone burned.\n"
    "- Stone bricks = 4 stone -> 4 stone bricks, at a crafting table.\n"
    "- Mine plenty of cobblestone: 8 go into the furnace and every stone costs one more.")


def rules_for(reward_cfg):
    """Observation RULES for the chain this ladder scores. Unknown ladder -> prefix only:
    serving a tail the task cannot walk is the bug this function exists to prevent."""
    # `reward_cfg or []` would raise on the numpy array parquet hands us ("truth value of an
    # array with more than one element is ambiguous"), so test for None explicitly.
    ids = set()
    for r in ([] if reward_cfg is None else reward_cfg):
        try:
            ids.add(str(dict(r).get("identity", "")))
        except Exception:
            pass
    if any(i.endswith("torch") for i in ids):
        return RULES_SHARED + RULES_TAIL_TORCH
    if any("stone_brick" in i for i in ids):
        return RULES_SHARED + RULES_TAIL_BRICKS
    if any("iron" in i for i in ids):
        return RULES_SHARED + RULES_TAIL_IRON
    return RULES_SHARED


WARNING_TEXT = "WARNING: no progress for 5+ turns - change depth or direction, or re-read RULES."
CONTRACT_TEXT = "Reply with <think>your reasoning</think> then <action>one short instruction</action>."

# ---- per-episode video dump (opt-in: MC_VIDEO_DUMP=<dir>) ------------------------------------
# Records the FULL render frame (info['pov'], RENDER_SIZE) for every sim frame of the first
# MC_VIDEO_SLOTS slots, one mp4 per episode, finalized as slot{i}_ep{n}_r{reward}.mp4.
# Frames come from the same info dict the sim already returns -- zero extra rendering cost
# beyond the encode. Guarded everywhere: video failure must never break a rollout.
VIDEO_DIR = os.environ.get("MC_VIDEO_DUMP")
VIDEO_SLOTS = int(os.environ.get("MC_VIDEO_SLOTS", "2"))
# store a PNG frame sequence (reliable but disk-hungry), assemble H.264 with ffmpeg later.
# See the notes in _vid_frame.
FRAME_DIR = os.environ.get("MC_VIDEO_FRAMES")
VIDEO_FPS = 20
# Agent labels are inferred from the instruction's verb; MC_DESK_LABELS=1 paints them onto the
# recorded video (off by default, so a recording never shows labels that were not decided).
DESK_LABELS = os.environ.get("MC_DESK_LABELS", "0") == "1"
if VIDEO_DIR:
    os.makedirs(VIDEO_DIR, exist_ok=True)

# ---- policy trajectory dump (opt-in: MC_TRAJ_DUMP=<dir>) -------------------------------------
# Written BY THE SERVER, one JSON line per finished episode, frames saved alongside. Authoritative:
# no post-hoc joining of separate logs (that is what produced a fabricated trajectory once before).
def _traj_cfg(env_key, file_name, default=None):
    """Env var first, then a marker file next to the run root.

    A server started inside a long-lived allocation is re-exec'd by its supervisor with the
    SHELL's environment, which cannot be changed from outside. The marker file lets an already
    allocated server be switched into trajectory recording by writing a file and restarting the
    process, instead of resubmitting the job and re-queueing for the node.
    """
    v = os.environ.get(env_key)
    if v:
        return v
    root = os.environ.get("MCROOT", "")
    if not root:                # no marker-file root configured: env var or default only
        return default
    try:
        with open(os.path.join(root, "rl", file_name)) as fh:
            v = fh.read().strip()
        return v or default
    except Exception:  # noqa: BLE001 - absent marker is the normal case
        return default


TRAJ_DIR = _traj_cfg("MC_TRAJ_DUMP", ".mc_traj_dump")
TRAJ_SLOTS = int(_traj_cfg("MC_TRAJ_SLOTS", ".mc_traj_slots", "2"))  # first N slots only (volume)
_EP_SEQ = [0]
# Frame filenames must be unique ACROSS SERVER PROCESSES. _EP_SEQ restarts at 0 every boot, so
# with a bare ep{id}_t{turn}.png every new server writing into the same MC_TRAJ_DUMP dir silently
# overwrote the previous generation's frames (traj.jsonl appends fine, so only the pixels were
# lost). Tag them with the pid.
_RUN_TAG = f"p{os.getpid()}"

def _traj_init():
    if TRAJ_DIR:
        os.makedirs(os.path.join(TRAJ_DIR, "frames"), exist_ok=True)
        print(f"[server] policy traj dump -> {TRAJ_DIR} (first {TRAJ_SLOTS} slots)", flush=True)

def _traj_save_frame(arr, name):
    from PIL import Image
    path = os.path.join(TRAJ_DIR, "frames", name)
    Image.fromarray(np.asarray(arr, dtype=np.uint8)).save(path)
    return os.path.join("frames", name)

def _traj_write(rec):
    with open(os.path.join(TRAJ_DIR, "traj.jsonl"), "a") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


# One STEVE-1 policy PER device. The whole env step is STEVE inference (measured: env
# step time ~= 0.326s x n_steve_actions, JVM floor ~0), so with every slot sharing one
# GPU the ~800 forwards/step (20 slots x 40 frames) serialize on a single compute stream.
# Placing slot i on cuda:(i % n_gpu) lets the idle GPUs on the node absorb that in parallel.
# 410M params (~1.6GB) per GPU is trivial. Keyed by device string so the cache is per-GPU.
# With MC_STEVE_URL set the single entry is a RemoteSteve proxy instead (steve_service.py).
_STEVE = {}
_SERVER_UID = uuid.uuid4().hex[:8]   # namespaces this process's slot ids at the remote service


# ---- Malmo port window per job (MC_MALMO_BASE_PORT) ------------------------------------------
# minerl picks each Minecraft instance's port as base(9000) + n_instances + (17*pid) % 3989 and
# only checks that the port is free at pick time; the JVM binds it a minute later. Two env
# servers on one node (several evaluation lanes) therefore race for the same ports and one JVM
# dies with "Address already in use". With MC_MALMO_BASE_PORT set, this server takes ports base,
# base+1, ... only -- give co-hosted servers disjoint windows (a launcher may derive the base from a
# scheduler job id). Unset -> stock minerl.
def _install_malmo_port_window():
    base = int(os.environ.get("MC_MALMO_BASE_PORT", "0") or 0)
    if not base:
        return
    try:
        from minestudio.simulator.minerl.env import malmo as _malmo
    except Exception as e:  # noqa: BLE001 - not fatal, the stock allocator still works
        print(f"[server] malmo port window not installed ({e})", flush=True)
        return

    def _get_valid_port(cls, _base=base):
        port = _base + cls.ninstances
        while cls._is_port_taken(port) or cls._port_in_instance_pool(port):
            port += 1
        return port

    _malmo.InstanceManager._get_valid_port = classmethod(_get_valid_port)
    print(f"[server] malmo port window: {base}..{base + 63}", flush=True)


_install_malmo_port_window()


class RemoteSteve:
    """Drop-in for SteveOnePolicy in Slot.step() when MC_STEVE_URL is set: the policy runs in
    steve_service.py (one GPU, frames of all slots batched per forward) and this process needs
    no GPU. prepare_condition() only stamps the instruction with a nonce (the service samples the
    text condition once per nonce, as the local path samples it once per step); get_action()
    ships the frame and returns (action, state token). The token IS the recurrent-state chain id
    at the service: state_in=None mints a new chain (exactly "start fresh" for the local policy),
    a token continues its own chain -- so a retired slot's orphan thread and its replacement
    never share a chain. Every call carries a request id: a retry after a lost reply returns the
    cached action instead of advancing the chain twice. The service URL is re-read from
    MC_STEVE_URL_FILE (if set) whenever a call fails, so a service that moved hosts is found."""

    def __init__(self, url, wait_s=None):
        import requests
        self.url = url.rstrip("/")
        self.url_file = os.environ.get("MC_STEVE_URL_FILE", "").strip() or None
        self._local = threading.local()
        self._requests = requests
        self.epoch = None
        self._warned = set()
        wait_s = float(os.environ.get("MC_STEVE_WAIT_S", "1800")) if wait_s is None else wait_s
        t0 = time.time()
        while True:
            try:
                h = requests.get(f"{self.url}/health", timeout=10).json()
                if h.get("ok") and "mean_batch" in h:    # a STEVE service, not some other /health
                    self.epoch = h.get("epoch")
                    print(f"[server] STEVE-1 remote at {self.url}: {h}", flush=True)
                    break
            except Exception:                            # noqa: BLE001 - service still starting
                pass
            if time.time() - t0 > wait_s:
                raise RuntimeError(f"STEVE-1 service {self.url} not healthy after {wait_s:.0f}s")
            self._refresh_url()
            time.sleep(10)

    def _refresh_url(self):
        if not self.url_file:
            return
        try:
            u = open(self.url_file).read().strip().rstrip("/")
        except OSError:
            return
        if u and u != self.url:
            print(f"[server] STEVE-1 service moved: {self.url} -> {u}", flush=True)
            self.url = u

    def _session(self):
        s = getattr(self._local, "s", None)
        if s is None:
            s = self._local.s = self._requests.Session()
        return s

    def prepare_condition(self, instruction, deterministic=False):
        text = instruction["text"]
        if isinstance(text, list):
            text = text[0]
        return {"text": str(text), "cond_scale": float(instruction["cond_scale"]),
                "nonce": uuid.uuid4().hex, "det": bool(deterministic)}

    def get_action(self, input, state_in=None, deterministic=False, **kw):
        img = input["image"]
        arr = img.detach().cpu().numpy() if hasattr(img, "detach") else np.asarray(img)
        arr = np.ascontiguousarray(arr.reshape(arr.shape[-3:]), dtype=np.uint8)   # (H, W, 3)
        cond = input["condition"]
        fresh = not isinstance(state_in, str)
        sid = f"{_SERVER_UID}:{uuid.uuid4().hex[:12]}" if fresh else state_in
        payload = {"sid": sid, "req": uuid.uuid4().hex[:16], "text": cond["text"],
                   "cond_scale": cond["cond_scale"], "nonce": cond["nonce"], "fresh": fresh,
                   "det": bool(deterministic or cond.get("det", False)),
                   "image": base64.b64encode(arr.tobytes()).decode(), "shape": list(arr.shape)}
        deadline = time.time() + float(os.environ.get("MC_STEVE_RETRY_S", "600"))
        last = None
        while True:
            try:
                r = self._session().post(f"{self.url}/act", json=payload, timeout=120)
                r.raise_for_status()
                out = r.json()
                if out.get("stale") and sid not in self._warned:
                    self._warned.add(sid)
                    print(f"[server] STEVE-1 service lost the chain {sid} (service epoch {out.get('epoch')} vs "
                          f"{self.epoch}): continuing from a fresh recurrent state", flush=True)
                if out.get("epoch") and out.get("epoch") != self.epoch:
                    print(f"[server] STEVE-1 service restarted (epoch {self.epoch} -> {out.get('epoch')})", flush=True)
                    self.epoch = out.get("epoch")
                return {k: np.asarray(v) for k, v in out["action"].items()}, sid
            except Exception as e:                       # noqa: BLE001 - service restart / hiccup
                last = e
                if time.time() > deadline:
                    raise RuntimeError(f"STEVE-1 service {self.url} failed for {sid}: {last}")
                self._refresh_url()
                time.sleep(2)


_STEVE_LOCKS = {}


def _steve_lock(model, device):
    """Serialises get_action() on a LOCAL policy: MaskedAttention.forward stores the attention
    mask on the shared module before running the block, so concurrent slot threads on one policy
    would silently run under each other's masks. The remote proxy needs no lock (each call is an
    independent HTTP request; the service serialises its own forwards)."""
    if isinstance(model, RemoteSteve):
        return contextlib.nullcontext()
    return _STEVE_LOCKS.setdefault(str(device), threading.Lock())


def load_steve(device="cuda"):
    url = os.environ.get("MC_STEVE_URL", "").strip()
    if url:
        if "remote" not in _STEVE:
            _STEVE["remote"] = RemoteSteve(url)
        return _STEVE["remote"]
    if os.environ.get("MC_STEVE_REQUIRED", "0") == "1" or not str(device).startswith("cuda"):
        raise RuntimeError("MC_STEVE_URL is unset on a CPU-only / MC_STEVE_REQUIRED env server -- "
                           "refusing to run STEVE-1 in-process")
    if device not in _STEVE:
        from minestudio.models import SteveOnePolicy
        repo = os.environ.get("STEVE1_CKPT", "CraftJarvis/MineStudio_STEVE-1.official")
        _STEVE[device] = SteveOnePolicy.from_pretrained(repo).to(device).eval()
        print(f"[server] STEVE-1 loaded ({repo}) on {device}", flush=True)
    return _STEVE[device]


def extract_instruction(text, max_words=20):
    vis = _THINK.sub("", text or "")
    m = _ACTION.search(vis)
    if not m:
        return None
    s = re.sub(r"\s+", " ", m.group(1)).strip()
    return " ".join(s.split()[:max_words]) if s else None


def _last_frame_b64(slot):
    """A slot's most recent frame, or black — used on the error paths so a frame is never missing."""
    try:
        if getattr(slot, "frame", None) is not None:
            return png_b64(slot.frame)
    except Exception:
        pass
    return png_b64(np.zeros((slot.obs_size[1], slot.obs_size[0], 3), dtype=np.uint8))


def png_b64(arr):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.asarray(arr, dtype=np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()



# ---- world snapshot support (counterfactual branches) -----------------------------------------
# Principle: the agent can only modify blocks near itself, and those cells were necessarily
# observed before being changed. So each cell's FIRST-seen value is the baseline, its last-seen
# value is the current one, and a difference between the two is one edit. A snapshot = seed +
# this diff + player state -- a ~KB JSON, no world files.
#
# Two experimentally confirmed traps; do not re-step on them:
#   * the voxels upper bound is an OPEN interval: requesting [-6,6] returns up to +5 only. If
#     the air-backfill loop is written range(lo, hi+1), the dx=+6/dz=+6 planes get recorded as
#     air every frame and the first player move mass-produces fake diffs (once inflated a diff
#     from 60 to 367 entries).
#   * voxels returns NON-AIR blocks only, so cells inside the box that did not come back must
#     be recorded as air explicitly, or "a block was mined out" is lost entirely.
#
# server_state: beyond world+player, the snapshot carries the slot's Python-side scoring
# state (see Slot.snapshot). Without it, a /restore'd branch inherits the TARGET slot's
# "which milestones already paid" memory: a milestone the parent already collected pays 0 on
# its second achievement, and the counterfactual branch is forced to route around the reward
# signal entirely. Restore ORDER is load-bearing: server_state must be applied only after both
# the world and player command bursts have settled, or the counter baselines read a dirty
# mid-restore state.
VOX_BOX = (-6, 6, -4, 5, -6, 6)
_LEGACY_SNAP_WARNED = [False]     # legacy snapshots (no server_state) warn once only

class _WorldTracker:
    __slots__ = ("baseline", "current")
    def __init__(self):
        self.baseline = {}; self.current = {}
    def observe(self, voxels, origin):
        if not voxels: return
        ox, oy, oz = origin
        seen = set()
        for e in voxels:
            if not isinstance(e, dict): continue
            k = (ox + e["x"], oy + e["y"], oz + e["z"])
            seen.add(k)
            self.baseline.setdefault(k, e.get("type"))
            self.current[k] = e.get("type")
        for dx in range(VOX_BOX[0], VOX_BOX[1]):          # open interval, do not +1
            for dy in range(VOX_BOX[2], VOX_BOX[3]):
                for dz in range(VOX_BOX[4], VOX_BOX[5]):
                    k = (ox + dx, oy + dy, oz + dz)
                    if k not in seen:
                        self.baseline.setdefault(k, "minecraft:air")
                        self.current[k] = "minecraft:air"
    def diff(self):
        return {k: v for k, v in self.current.items() if self.baseline.get(k) != v}

def _blk(p):
    import math
    return (math.floor(p[0]), math.floor(p[1]), math.floor(p[2]))   # negative coords need floor


class Slot:
    """One Minecraft world + its STEVE-1 recurrent state."""

    def __init__(self, idx, frames_per_action, obs_size, device, cond_scale=4.0):
        self.idx = idx; self.K = frames_per_action; self.obs_size = obs_size
        self.device = device; self.cond_scale = cond_scale
        self.sim = None; self.state = None; self.frame = None
        self.task = ""; self.cum = 0.0; self.n = 0; self.max_n = 8; self.ds = "minecraft"
        self.rules = RULES_SHARED
        self.ladder = _LADDER_IRON
        self._tracker = None; self._seed = None
        self._snap_shift = {}    # branch total correction; {} on non-branch paths (see _totals)
        # One lock per slot, held across every per-slot operation (reset/step/step_slot/
        # snapshot/restore/cmd). Without it, counterfactual branch traffic (/step_slot,
        # /restore) can interleave with the mainline /step fanout inside the SAME slot:
        # /step dispatches per-slot work to threads, so a /restore arriving mid-step would
        # issue /setblock bursts while step() is mid-tick -- the JVM serializes commands,
        # but the Python-side bookkeeping (self.n, _winfo, _prev_snap, tracker) would be
        # torn between the two writers. The lock makes each slot's ops atomic; different
        # slots still run fully in parallel.
        self._op_lock = threading.Lock()
        # Set when a slot misses its deadline. Waiting the full deadline again on EVERY later
        # call turns a 3s step into a 300s one: the transfer eval measured exactly 300.1s per
        # step, 120 turns of that being 10 hours for one episode. A slot that hung once is not
        # coming back without a successful reset, so stop paying for it.
        self._timed_out = False


    # ---- counterfactual branches: snapshot & restore ----------------------------------------
    def snapshot(self):
        """World + player + Python-side scoring state (server_state), one ~KB JSON. No world
        files: terrain is seed-deterministic, so it suffices to remember which blocks changed
        relative to a fresh same-seed world."""
        if self._tracker is None:
            return {"error": "MC_SNAPSHOT not enabled"}
        info = getattr(self, "_winfo", {}) or {}
        ls = info.get("location_stats") or {}
        def g(k):
            v = ls.get(k)
            try: return float(v.item() if hasattr(v, "item") else v)
            except Exception: return None
        hand = (info.get("equipped_items") or {}).get("mainhand")
        if isinstance(hand, dict): hand = hand.get("type")
        hand = str(hand) if hand not in (None, "none", "air", "minecraft:air") else None
        diff = self._tracker.diff()
        snap = {"seed": self._seed, "build_seed": getattr(self, "_build_seed", None),
                "turn": self.n, "cum": self.cum,
                "blocks": {f"{x},{y},{z}": t for (x, y, z), t in diff.items()},
                # blocks_base: the pre-change value of every diffed cell (the tracker's
                # first-seen reading). restore() needs it twice: to revert the TARGET slot's
                # own stray edits back to terrain, and to seed the branch's fresh tracker so a
                # snapshot taken FROM the branch still carries the parent's edits as diff
                # (without it, replayed parent /setblock edits melt into the new baseline and
                # nested branching silently loses world state).
                "blocks_base": {f"{x},{y},{z}": self._tracker.baseline.get((x, y, z))
                                for (x, y, z) in diff},
                # Inventory is captured UNCONDITIONALLY. It used to be gated on
                # MC_SCRIPTED_CRAFT while restore() unconditionally issued /clear @p, so with
                # scripted crafting off (the default) every restored branch silently lost its
                # whole inventory -- and with it any consistency between reward_memory ("you
                # already scored obtain_wood") and the world ("you hold no wood").
                "player": {"x": g("xpos"), "y": g("ypos"), "z": g("zpos"),
                           "yaw": g("yaw"), "pitch": g("pitch"),
                           "hand": hand, "inv": self._inv_now()}}
        # server_state: the Python-side scoring state. reward_memory is the core of
        # double-count/zero-count correctness: RewardsCallback and _scripted_reward share this
        # one dict for max_reward_times capping. prev_snap stores the PARENT-view cumulative
        # totals _totals() (this slot's counters + branch correction); a restore must never
        # copy it verbatim (per-slot counters are independent) -- it is only used to compute
        # the _snap_shift delta, see _apply_server_state. Nested branching is load-bearing
        # here: if this slot is itself a restored branch (_snap_shift non-empty), the bare
        # _prev_snap is the scratch slot's local count and would silently drop everything
        # accumulated before the first branch point -- a grandchild branch's "Totals so far"
        # and traj final_totals would collapse to the scratch slot's own ledger; on
        # non-branch slots _totals() == dict(_prev_snap), byte-for-byte.
        rm = getattr(getattr(self, "reward_cb", None), "reward_memory", None) or {}
        snap["server_state"] = {
            "reward_memory": {str(k): int(v) for k, v in dict(rm).items()},
            "reward_cfg": getattr(self, "_rc", None),
            "task": self.task, "ds": self.ds, "max_n": self.max_n,
            "hist": list(getattr(self, "_hist", []) or []),
            "prev_snap": self._totals(),
            "spawn_xz": (list(self._spawn_xz)
                         if getattr(self, "_spawn_xz", None) is not None else None),
        }
        return snap

    def restore(self, snap, build=False):
        """Two-phase restore. ORDER AND WAITS ARE LOAD-BEARING: commands land over several
        ticks, and firing the block and player bursts together lets /tp collide on the same
        tick with a not-yet-effective `/setblock ... minecraft:air` -- the player teleports
        into solid stone -> dies -> the env auto-resets.
        Measured (world_engine E5f/E5g): 44 commands in one burst -> ticks 1-4 fully correct,
        tick 5 position AND inventory zero out together; two phases -> all 25 ticks
        done=False, stable coordinates, 20/20 health, 5/5 snapshot blocks correct.

        build=True (branch slots): the slot is ALWAYS put through a seeded hard reset into
        the snapshot's build_seed (+ server_state task/reward cfg) before the restore is
        applied; a JVM is built only if the slot never hosted one. The first production
        preference iteration died 4/4 on spare slots 8/9 having no sim (the mainline batch
        only ever resets slots [0, TBS)) and every /restore answered 'no sim'. Without the
        flag the strict old semantics hold, so a mainline restore can never silently land on
        a rebuilt wrong-world slot, and the seed-mismatch refusal in _apply_server_state
        (and its smoke check) is untouched."""
        if build:
            want = snap.get("build_seed", snap.get("seed"))
            if want is None:
                return {"ok": False, "error": "build requested but snapshot carries no seed"}
            # ALWAYS a seeded hard reset into build_seed (JVM kept: seeding needs no rebuild,
            # see _reset_sim) -- never "reuse the slot because _build_seed already matches":
            #  (1) world identity: the old shortcut assumed a JVM built from seed S still
            #      hosted world S, but MineRL seeds the FIRST reset only, so the slot's later
            #      episodes were unseeded and the branch restored a forest snapshot into a
            #      sand hole (a branch world built from an earlier episode's seed). Only a
            #      fresh seeded reset
            #      yields the snapshot's terrain.
            #  (2) slot residue: even in the right world a reused spare slot carries the
            #      previous branch's block edits (the revert pass below found 594 own edits
            #      outside the parent diff), phantom voxel diffs in its tracker, a non-zero
            #      selected hotbar slot (the weapon.mainhand backstop write below then lands
            #      in THAT slot and duplicates the held item) and its STEVE recurrent state.
            #      A fresh episode zeroes all of it; the revert pass then finds nothing.
            ss = snap.get("server_state") or {}
            t0 = time.time()
            self.reset({"task_text": ss.get("task") or "Explore.",
                        "data_source": ss.get("ds") or "minecraft",
                        "max_high_steps": ss.get("max_n") or 8,
                        "reward_cfg": ss.get("reward_cfg") or [],
                        "seed": int(want)})
            print(f"[server] slot{self.idx} built branch world seed={want} "
                  f"in {time.time()-t0:.1f}s", flush=True)
            # the episode-seed LABEL must match the snapshot lineage: _apply_server_state
            # compares snap['seed'] against it, and snapshots taken FROM this branch must
            # carry the parent's label onward
            self._seed = snap.get("seed")
        if self.sim is None:
            return {"ok": False, "error": "no sim"}
        # The branch is NOT the episode this slot was recording: close the target slot's
        # still-open trajectory record / video writer under the original episode first.
        # Without the close, branch turns get appended into the target slot's own episode
        # record/mp4, splicing together a "trajectory" nobody ever ran -- and trajectories
        # are the authoritative post-scoring verification channel (scores are audited against
        # them), so contamination is worse than absence. Must close BEFORE overwriting
        # self.n/self.cum: _vid_close names the finished file with self.cum, and the closed
        # traj record writes the target slot's own ledger up to this moment.
        self._vid_close()
        if getattr(self, "_traj", None) is not None:
            self._traj["total_reward"] = self.cum
            self._traj["final_totals"] = self._totals()
            self._traj["truncated_by_restore"] = True
            _traj_write(self._traj); self._traj = None
        blocks = snap.get("blocks") or {}
        # Revert THIS slot's own stray edits first. The parent diff only states what the
        # parent changed; a spare slot that already ran an earlier branch has edits of its
        # own at positions the parent never touched, and without this pass the two branch
        # slots would start from DIFFERENT worlds (branch comparability is the whole point).
        # The revert target is this slot's own tracker baseline -- its first-seen value for
        # the cell, i.e. natural same-seed terrain in every non-pathological case.
        if self._tracker is not None:
            own = self._tracker.diff()
            reverted = 0
            for (x, y, z), _cur in own.items():
                key = f"{x},{y},{z}"
                if key in blocks:
                    continue                  # the parent diff will overwrite it anyway
                base = self._tracker.baseline.get((x, y, z))
                if base:
                    self.sim.env.execute_cmd(f"/setblock {x} {y} {z} {base}")
                    reverted += 1
            if reverted:
                print(f"[server] slot{self.idx} restore: reverted {reverted} own edits "
                      f"outside the parent diff", flush=True)
        for k, t in blocks.items():
            self.sim.env.execute_cmd(f"/setblock {k.replace(',', ' ')} {t}")
        self._idle(12)                       # let the blocks land before placing the player
        p = snap.get("player") or {}
        inv = dict(p.get("inv") or {})
        hand = p.get("hand")
        hk = None
        if hand:
            b = hand.split(":")[-1]
            hk = b if b in inv else (hand if hand in inv else None)
        cmds = ["/clear @p"]
        if None not in (p.get("x"), p.get("y"), p.get("z")):
            cmds.append(f"/tp @p {p['x']:.2f} {p['y']:.2f} {p['z']:.2f} "
                        f"{(p.get('yaw') or 0.0):.1f} {(p.get('pitch') or 0.0):.1f}")
        # CANONICALIZATION for branch comparability. Health, food, world time and weather are
        # not captured in the snapshot (no command channel reads them back), so two branch
        # slots would otherwise inherit their own residual values -- slot M half-dead from a
        # previous pair's fall while slot N is at 20/20, or one world at night while the
        # other is at noon -- and the branch comparison would be scored on slot history, not
        # on the counterfactual action. Forcing fixed canonical values makes every restored
        # branch identical on these axes (to each OTHER, which is what pair scoring needs;
        # the parent's exact time is deliberately not reproduced). 1.16.5 command syntax.
        # NOT canonicalized and known residuals: entity state (mobs, dropped items) and
        # block-entity contents (furnace/chest inventories). The scripted crafting ladder
        # never uses placed containers, so the practical residual is mob positions -- an env
        # stochasticity on the same footing as post-branch world randomness.
        cmds.append("/effect give @p minecraft:instant_health 1 9 true")
        cmds.append("/effect give @p minecraft:saturation 1 9 true")
        cmds.append("/time set 6000")
        cmds.append("/weather clear")
        # The held item goes to hotbar.0 (a fresh episode selects slot 0), with one final
        # weapon.mainhand write as a backstop. Not restoring the main hand has very real
        # consequences: coordinates/inventory/health self-checks all green while the agent
        # pounds a wall holding cobblestone -- 800 ticks, zero blocks mined.
        # Counts are chunked at 64 per slot: /replaceitem silently CLAMPS a count above the
        # stack limit, so writing {"cobblestone": 130} as one command restored 64 and lost
        # the other 66 -- the branch then diverges from reward_memory's idea of what it owns.
        # (Items with a smaller stack limit than 64 still clamp; the tasks in play stack 64.)
        order = ([hk] if hk else []) + sorted(k for k in inv if k != hk)
        chunks = []                                    # (item, count<=64), hand item first
        for it in order:
            q = int(inv[it])
            while q > 0:
                chunks.append((it, min(q, 64)))
                q -= 64
        slots_ = [f"hotbar.{i}" for i in range(9)] + [f"inventory.{i}" for i in range(27)]
        if len(chunks) > len(slots_):
            print(f"[server] slot{self.idx} restore: inventory needs {len(chunks)} slots, "
                  f"only {len(slots_)} exist -- dropping the overflow", flush=True)
        for i, (it, q) in enumerate(chunks[:len(slots_)]):
            cmds.append(f"/replaceitem entity @p {slots_[i]} minecraft:{it} {q}")
        if hk:
            cmds.append(f"/replaceitem entity @p weapon.mainhand minecraft:{hk} "
                        f"{min(int(inv[hk]), 64)}")
        for c in cmds:
            self.sim.env.execute_cmd(c)
        self._idle(8)
        self.n = int(snap.get("turn") or 0); self.cum = float(snap.get("cum") or 0.0)
        # The branch's fresh tracker is SEEDED with the parent diff (baseline = the parent's
        # pre-change cell values from blocks_base, current = the replayed values), so a
        # snapshot taken FROM this branch reports the parent's edits as part of its own diff.
        # This closes the old TODO: previously the replayed /setblock edits melted into the
        # fresh tracker's baseline and nested branching silently lost world state. Legacy
        # snapshots without blocks_base fall back to the old behavior (empty seed), which
        # only matters if someone re-snapshots from a branch of a pre-upgrade snapshot.
        nt = _WorldTracker()
        base_map = snap.get("blocks_base") or {}
        for k, t in blocks.items():
            pos = tuple(int(v) for v in k.split(","))
            b = base_map.get(k)
            if b and b != t:
                nt.baseline[pos] = b
                nt.current[pos] = t
        self._tracker = nt                   # branch diffs restart here, seeded with the parent diff
        self._snap_shift = {}                # the old timeline's total correction dies with it
        # Apply the Python-side scoring state only after world/player settled (order
        # is load-bearing, see the module header). Effective only with MC_SNAPSHOT=1 and a
        # snapshot that actually carries server_state -- legacy snapshots keep restoring
        # world+player only.
        applied = False
        st = snap.get("server_state")
        if os.environ.get("MC_SNAPSHOT", "0") == "1":
            if isinstance(st, dict):
                applied = self._apply_server_state(st, snap)
            elif not _LEGACY_SNAP_WARNED[0]:
                _LEGACY_SNAP_WARNED[0] = True
                print("[server] restore: snapshot has no server_state (older snapshot format) -- "
                      "restoring world+player only; scoring state stays the target slot's",
                      flush=True)
        # The branch is its own episode: reopen the trajectory and video channels the way
        # reset() does (both no-ops when the channels are disabled). _apply_server_state's
        # _noop_tick already refreshed _winfo into wrapped info (with pov); on the legacy
        # snapshot path _winfo is the bare info _idle stored, and _vid_start gives up on its
        # own when pov is missing.
        self._vid_start(getattr(self, "_winfo", {}) or {})
        if TRAJ_DIR and self.idx < TRAJ_SLOTS:
            _EP_SEQ[0] += 1
            self._ep_id = _EP_SEQ[0]
            f = _traj_save_frame(self.frame, f"{_RUN_TAG}_ep{self._ep_id:05d}_t00.png")
            self._traj = {"episode": self._ep_id, "slot": self.idx, "task": self.task,
                          "seed": snap.get("seed"), "max_high_steps": self.max_n,
                          "data_source": self.ds, "start_frame": f, "turns": [],
                          # mark which parent turn this record branched from, so trajectory
                          # readers can line it up with the mainline
                          "branch_of_turn": snap.get("turn")}
        return {"ok": True, "blocks": len(blocks), "cmds": len(cmds), "server_state": applied}

    def _apply_server_state(self, st, snap):
        """Apply the snapshot's Python-side scoring state to this slot, making the branch
        scoring-identical to the parent at the branch point.

        Must be called after the world/player restore has settled: counter baselines must
        read the POST-restore wrapped info. Two things are deliberately NOT restored
        (touching them would be wrong):
          * reward_cb.prev_info -- RewardsCallback's diff baseline tracks THIS slot's sim's
            absolute counters (the restore commands /setblock /tp /clear /replaceitem never
            touch stat counters, so it is consistent by construction); copying the parent's
            absolute readings would make the first diff invent or swallow events.
          * _prev_snap's absolute values -- likewise re-anchored to this slot's current
            readings; the parent-view cumulative totals are reconstructed at render time via
            the _snap_shift delta (_totals).

        Returns True when everything applied; a failed counter rebase returns False, which
        restore() writes into the /restore reply's server_state field -- the driver side then
        rules the branch restore_failed."""
        if snap.get("seed") is not None and self._seed is not None \
                and snap.get("seed") != self._seed:
            # HARD failure, not a warning: the block diff is meaningful only relative to the
            # same-seed terrain. Replaying it onto a different world "succeeds" command-wise
            # while producing a state no timeline ever contained; a branch scored on it is
            # noise. Returning False makes /restore answer server_state:false, which the
            # driver's _restore_failed treats as a dead branch.
            print(f"[server] slot{self.idx} restore: SEED MISMATCH (snap {snap.get('seed')} "
                  f"vs slot {self._seed}) -- refusing server_state", flush=True)
            return False
        # milestone cap memory: shared by RewardsCallback and _scripted_reward; both
        # double-count and zero-count behavior hang off it
        rm = st.get("reward_memory")
        if rm is not None and getattr(self, "reward_cb", None) is not None:
            self.reward_cb.reward_memory = {str(k): int(v) for k, v in dict(rm).items()}
        rc = st.get("reward_cfg")
        if rc is not None:
            if rc != getattr(self, "_rc", None) and getattr(self, "reward_cb", None) is not None:
                self.reward_cb.reward_cfg = rc     # same swap as reset(): never rebuild the JVM
            self._rc = rc
        if st.get("task") is not None: self.task = st["task"]
        if st.get("ds") is not None: self.ds = st["ds"]
        if st.get("max_n") is not None: self.max_n = int(st["max_n"])
        self._hist = list(st.get("hist") or [])
        sx = st.get("spawn_xz")
        self._spawn_xz = (float(sx[0]), float(sx[1])) if sx else None
        self._obs_text_before = ""
        # STEVE recurrent state is GPU tensors and cannot cross the JSON channel; zero it
        # uniformly so every branch at least starts from the same deterministic state
        self.state = None
        # Re-anchor the counter baseline to this slot's sim's current readings; the delta to
        # the parent totals goes into _snap_shift. _idle stores bare info in _winfo, so
        # getting wrapped counters takes one _noop_tick (which also refreshes the frame).
        want = dict(st.get("prev_snap") or {})
        try:
            self._noop_tick()
            live = self._snap(self._winfo)
            self._prev_snap = live
            self._snap_shift = {k: want.get(k, 0) - live.get(k, 0)
                                for k in set(want) | set(live)
                                if want.get(k, 0) != live.get(k, 0)}
        except Exception:
            traceback.print_exc()
            # A failure must not fake green (a bad reading is not evidence): if _prev_snap
            # kept the target slot's old baseline, the branch's first _delta would invent or
            # swallow events against the wrong baseline. Blank it + return False so the
            # /restore reply carries server_state:false and the driver (_restore_failed)
            # rules this branch out instead of running on with a broken ledger.
            self._prev_snap = {}
            print(f"[server] slot{self.idx} restore: counter rebase FAILED -- "
                  f"reporting server_state=false", flush=True)
            return False
        return True

    def _idle(self, n):
        """Run n no-op ticks so already-issued commands take effect one by one."""
        import numpy as _np
        for _ in range(n):
            a = self.sim.env.noop_action()
            if self._tracker is not None:
                a["voxels"] = _np.array(VOX_BOX, dtype=_np.int32)
            try:
                obs, _r, _d, info = self.sim.env.step(a)
                self._winfo = info
            except Exception:
                break

    def _build(self, reward_cfg, biome, inventory, seed):
        # minerl's SOCKTIME (default 120s) caps FIRST-reset world generation; dense-tree biomes
        # (birch/dark forest) need ~130-200s on these nodes, so every slot died at gen time.
        # MC_SOCKTIME raises the module constant before any socket is created.
        _st = float(os.environ.get("MC_SOCKTIME", "0"))
        if _st > 0:
            from minestudio.simulator.minerl.env import _multiagent
            _multiagent.SOCKTIME = _st
        from minestudio.simulator import MinecraftSim
        from minestudio.simulator.callbacks import RewardsCallback
        # keep a handle: the reward spec is re-pointed in place on later episodes so the JVM is
        # built exactly ONCE per slot (rebuilding cost ~25s each and could hang the whole batch)
        self.reward_cb = RewardsCallback(reward_cfg or [])
        cbs = [self.reward_cb]
        # MinecraftSim accepts an inventory parameter, but PASSING IT ALONE DOES NOTHING --
        # InitInventoryCallback must be attached too, or it is silently ignored (E10 got three
        # empty runs in a row before this was located). The format is a list of slot dicts:
        # [{"slot": 0, "type": "oak_planks", "quantity": 64}]; an {item: count} dict raises
        # nothing and has no effect. Attached only when the caller actually specifies an
        # inventory, so no existing usage changes.
        if inventory:
            try:
                from minestudio.simulator.callbacks import InitInventoryCallback
                cbs.append(InitInventoryCallback(inventory))
            except Exception as e:
                print(f"[server] InitInventoryCallback attach failed: {e}", flush=True)
        kw = dict(action_type="agent", obs_size=self.obs_size, render_size=RENDER_SIZE,
                  seed=int(seed), callbacks=cbs)
        if biome: kw["preferred_spawn_biome"] = biome
        if inventory: kw["inventory"] = inventory
        # The WORLD identity of the episode this JVM hosts next. MinecraftSim.__init__ is the
        # ONLY place MineStudio calls env.seed(), and MineRL consumes that seed on the first
        # reset only (see _reset_sim), so this value is right for the construction episode
        # alone: _reset_sim re-seeds every later episode and overwrites it with the seed
        # actually used. Snapshots publish it as build_seed so a branch restore can
        # regenerate the matching world on a spare slot (see restore(build=True)).
        self._build_seed = int(seed)
        return MinecraftSim(**kw)

    # A JVM can die mid-run: one slot reset fine 19 times, then every subsequent reset hit
    # minerl's 120s socket timeout while the other 7 slots stayed at ~8s. That is a dead instance, not
    # heap pressure (GC/OOM count stayed 0, and the timing was 120.3s every time — a constant, not a
    # curve). Left alone it poisons 1/8 of every batch AND, since /reset is sequential, taxes every
    # training step by 120s. So: time each reset, and rebuild the slot's sim once it looks dead.
    # ...but that 60 s threshold was calibrated when a reset REUSED the slot's world and took
    # ~8 s. Seeding every episode (see _reset_sim) makes a healthy reset regenerate the world:
    # measured 145-275 s with 32 JVMs building across two servers on one node. Every healthy
    # reset then read as "slow", the streak reached 2, and the server rebuilt a perfectly good
    # JVM -- which is slower still, so rounds never completed and the client timed out at 1200 s.
    # Timing no longer discriminates (a dead JVM's 120 s socket timeout is now FASTER than a
    # healthy seeded reset), so the threshold only catches a genuinely stuck slot and the
    # reliable failure signal is the exception path below.
    SLOW_RESET_S = float(os.environ.get("MC_SLOW_RESET_S", "600"))

    def _reset_sim(self, kw, _rebuilt=False):
        # Seed THIS episode's world. MineRL consumes env._seed exactly once: reset() appends
        # it to the mission token only `if self._seed is not None` and its finally-block then
        # sets `self._seed = None` ("We don't force the same seed every episode, you gotta
        # send it yourself", minestudio/simulator/minerl/env/_multiagent.py), and
        # MinecraftSim.__init__ (`self.env.seed(seed)`, simulator/entry.py) is the only place
        # MineStudio ever calls it. So without this line every episode after a slot's first
        # one -- mainline /reset and branch rebuilds alike -- was generated UNSEEDED (random
        # world) while build_seed still advertised the construction seed, and a branch
        # restored a forest snapshot into a sand hole. env.seed() only stores the int and
        # seeds the gym spaces (no socket round-trip, no JVM rebuild); it runs before t0 so
        # it cannot feed the dead-JVM timing heuristic below. No seed in kw -> 0, the same
        # default _build uses, so the world stays reproducible.
        seed = kw.get("seed")
        seed = 0 if seed is None else int(seed)
        try:
            self.sim.env.seed(seed)
        except Exception as e:  # noqa: BLE001
            # loud, never fatal: a wrapper without the MineRL .env handle must not kill the
            # episode, but a silent miss would restore the unseeded-world bug invisibly
            print(f"[server] slot{self.idx} SEEDING FAILED ({type(e).__name__}: {e}) -- this "
                  f"episode's world is NOT reproducible and branch restores will not match",
                  flush=True)
        t0 = time.time()
        try:
            obs, info = self.sim.reset()
            # the world this JVM now hosts: what /snapshot publishes as build_seed
            self._build_seed = seed
            if os.environ.get("MC_SNAPSHOT", "0") == "1":
                # new world -> new tracker. The baseline must start HERE; carrying the
                # previous episode's over would count that whole episode's edits into the diff.
                self._tracker = _WorldTracker(); self._seed = kw.get("seed")
            slow = (time.time() - t0) > self.SLOW_RESET_S
        except Exception as e:
            print(f"[server] slot{self.idx} reset raised ({e})", flush=True)
            obs, info, slow = None, None, True
        if not slow:
            self._slow_streak = 0
            return obs, info
        self._slow_streak = getattr(self, "_slow_streak", 0) + 1
        print(f"[server] slot{self.idx} slow/failed reset "
              f"({time.time()-t0:.1f}s, streak {self._slow_streak})", flush=True)
        # one slow reset can be luck; two in a row means the JVM is gone — rebuild it once
        if self._slow_streak < 2 or _rebuilt:
            if obs is None:
                raise RuntimeError(f"slot{self.idx} reset failed and rebuild already attempted")
            return obs, info
        print(f"[server] slot{self.idx} REBUILDING dead JVM", flush=True)
        try: self.sim.close()
        except Exception: pass
        self.sim = self._build(self._rc, kw.get("spawn_biome"), kw.get("inventory"), seed)
        self._slow_streak = 0
        return self._reset_sim(kw, _rebuilt=True)

    def reset(self, kw):
        self.task = kw.get("task_text", "Explore.")
        self.ds = kw.get("data_source", "minecraft")
        self.rules = rules_for(kw.get("reward_cfg"))
        self.ladder = _ladder_for(kw.get("reward_cfg"))
        self.max_n = int(kw.get("max_high_steps", 8))
        # a slot keeps its JVM across episodes (rebuilding is very slow); reward cfg is per-episode,
        # so rebuild only when the reward spec actually changes
        rc = kw.get("reward_cfg") or []
        if self.sim is None:
            self.sim = self._build(rc, kw.get("spawn_biome"), kw.get("inventory"), kw.get("seed", 0))
        elif getattr(self, "_rc", None) != rc:
            # NEVER rebuild the sim for a new reward spec — swap it into the live callback instead
            self.reward_cb.reward_cfg = rc
            try: self.reward_cb.reward_memory = {}
            except Exception: pass
        self._rc = rc
        # Re-anchor "N blocks from where you started" on this episode's first frame:
        # _world_block sets _spawn_xz only while it is None, so without clearing it here every
        # later episode measured distance from the slot's FIRST-ever spawn (8143 blocks at
        # decision 5). Branch restores still inherit the parent's anchor: _apply_server_state
        # overwrites it from the snapshot after this reset.
        self._spawn_xz = None
        obs, info = self._reset_sim(kw)
        self.state = None; self.frame = obs["image"]; self.cum = 0.0; self.n = 0
        self._winfo = info if isinstance(info, dict) else {}
        # history + the post-reset counter baseline (CommandsCallback /give fires during
        # after_reset, so anything granted at spawn must NOT be reported as "you obtained it")
        self._hist = []
        self._prev_snap = self._snap(info)
        self._snap_shift = {}      # new episode: the previous branch timeline's correction is void
        self._vid_close()          # a resume/reset mid-episode must not leak an open writer
        self._vid_start(info)
        # begin a trajectory record for this episode
        self._obs_text_before = ""
        self._traj = None
        if TRAJ_DIR and self.idx < TRAJ_SLOTS:
            _EP_SEQ[0] += 1
            self._ep_id = _EP_SEQ[0]
            f = _traj_save_frame(self.frame, f"{_RUN_TAG}_ep{self._ep_id:05d}_t00.png")
            self._traj = {"episode": self._ep_id, "slot": self.idx, "task": self.task,
                          # record the world seed: without it a recorded episode cannot be replayed
                          # in the same world, which forces any video to be a re-execution
                          "seed": kw.get("seed"), "max_high_steps": self.max_n,
                          "data_source": self.ds, "start_frame": f, "turns": []}
        _rinfo = {"data_source": self.ds, "won": False}
        if SCRIPTED:
            _rinfo["inventory"] = self._inv_now()
        return {"text": self._text("(episode start)"), "image_b64": png_b64(self.frame),
                "info": _rinfo}

    def step(self, action_text):
        import torch
        self._obs_text_before = self._text('(pending)')
        instr = extract_instruction(action_text)
        self.n += 1
        if instr is None:
            done = self.n >= self.max_n
            self._log_turn("(malformed action)", 0.0, {})
            if getattr(self, "_traj", None) is not None:
                self._traj["turns"].append({"turn": self.n, "raw_action": (action_text or "")[:800],
                                            "instruction": None, "valid": False, "reward": 0.0,
                                            "frame": None, "events": {}})
                if done: _traj_write(self._traj); self._traj = None
            if done: self._vid_close()
            return {"text": self._text("INVALID: you must emit <action>...</action>"),
                    "reward": 0.0, "done": done, "image_b64": png_b64(self.frame),
                    "info": {"data_source": self.ds, "won": self.cum > 0, "is_action_valid": False,
                             "goal_reached": self._goal_reached()}}
        if SCRIPTED:
            sc = _match_scripted(instr)
            if sc is not None:
                return self._scripted_step(sc[0], sc[1], instr)
        model = load_steve(self.device)
        cond = model.prepare_condition({"cond_scale": self.cond_scale, "text": instr})
        r_sum = 0.0; events = {}; term = trunc = False; degraded = False
        obs = {"image": self.frame}; last_info = {}
        for _ in range(self.K):
            img = torch.from_numpy(np.asarray(obs["image"], dtype=np.uint8)).unsqueeze(0).unsqueeze(0).to(self.device)
            with _steve_lock(model, self.device), torch.no_grad():
                act, self.state = model.get_action({"image": img, "condition": cond},
                                                   state_in=self.state, deterministic=False)
            act = {k: (v.detach().cpu().numpy().squeeze() if hasattr(v, "detach") else v) for k, v in act.items()}
            if self._tracker is not None:
                act["voxels"] = np.array(VOX_BOX, dtype=np.int32)   # piggybacks, costs no extra tick
            obs, r, term, trunc, info = self.sim.step(act)
            if self._tracker is not None:
                _ls = info.get("location_stats") or {}
                try:
                    _o = _blk((float(_ls.get("xpos")), float(_ls.get("ypos")), float(_ls.get("zpos"))))
                    self._tracker.observe(info.get("voxels"), _o)
                except Exception:
                    pass
            # PHANTOM-REWARD GUARD. When a slot's JVM hangs, minerl hits its 120s socket timeout and
            # returns a RANDOM observation ("Terminating episode and sending random observation, be
            # aware. ... check to see if 'error' in info"). The stat counters in that degraded info
            # are garbage, so RewardsCallback diffs them against prev_info and fires EVERY stage at
            # once -- observed as reward_max=36 (= 2+3+3+4+8+16) on steps that took 123.6s.
            # Per-turn rewards are normally 1-3, so one phantom +36 dominates the advantage of
            # the whole batch. Never let it into the reward.
            if isinstance(info, dict) and "error" in info:
                print(f"[server] slot{self.idx} DEGRADED STEP (minerl error='{info.get('error')}') "
                      f"-- discarding reward {r} and ending the episode", flush=True)
                degraded = True
                break
            r_sum += float(r)
            if isinstance(info, dict) and "mine_block" in info:
                events = info["mine_block"]
            last_info = info
            self._vid_frame(info, instr)
            if term or trunc: break
        # second guard: no legitimate single turn can pay more than the largest single stage,
        # because every stage has max_reward_times=1. Anything above that is a counter glitch.
        _cap = max([float(c.get("reward", 0)) for c in (self._rc or [])] or [1e9])
        if r_sum > _cap:
            print(f"[server] slot{self.idx} IMPLAUSIBLE turn reward {r_sum} > max single stage "
                  f"{_cap} -- treating as a counter glitch, zeroing", flush=True)
            r_sum = 0.0; degraded = True
        self.frame = obs["image"]; self.cum += r_sum
        if isinstance(last_info, dict) and last_info:
            self._winfo = last_info
        delta = {} if degraded else self._delta(last_info)          # what actually changed this turn, all categories
        self._log_turn(instr, r_sum, delta)
        done = bool(term or trunc) or degraded or self.n >= self.max_n
        if getattr(self, "_traj", None) is not None:
            f = _traj_save_frame(self.frame, f"{_RUN_TAG}_ep{self._ep_id:05d}_t{self.n:02d}.png")
            self._traj["turns"].append({"turn": self.n, "raw_action": (action_text or "")[:800],
                                        "instruction": instr, "valid": True, "reward": r_sum,
                                        "frame": f, "delta": delta, "obs_text": self._obs_text_before,
                                        # delta covers mine_block/craft_item/custom/pickup; `events`
                                        # is mine_block only and is kept for backward compat. The
                                        # traj dump was BLIND to craft_item/custom before this, which
                                        # made per-stage attribution impossible on a tech-tree task.
                                        "events": {k: int(v) for k, v in list(events.items())[:8]}})
            if done:
                self._traj["total_reward"] = self.cum
                self._traj["final_totals"] = self._totals()
                _traj_write(self._traj); self._traj = None
        if done: self._vid_close()
        # "won" = any reward so far (the trainer's success_rate); "goal_reached" = the task's final
        # stage (the last reward_cfg entry, the iron pickaxe) has paid out
        _sinfo = {"data_source": self.ds, "won": self.cum > 0, "is_action_valid": True,
                  "goal_reached": self._goal_reached(),
                  "instruction": instr, "delta": delta,
                  "events": {k: int(v) for k, v in list(events.items())[:8]}}
        if SCRIPTED:
            _sinfo["inventory"] = self._inv_now()
        # P1/P2: no reward number in the text; the Last line reports what actually dropped,
        # rendered by the same rule as the history lines, plus the P6 failure hint.
        return {"text": self._text(f"executed '{instr}' -> got {self._got_str(delta)}"
                                   f"{self._hint(instr, delta)}"),
                "reward": r_sum, "done": done, "image_b64": png_b64(self.frame),
                "info": _sinfo}

    # ---- scripted crafting (see SCRIPTED above) --------------------------------------------------
    def _exec_cmd(self, cmd):
        """Run a Minecraft command on the live sim (costs one tick) and refresh wrapped obs/info.
        Same pattern CommandsCallback/InitInventoryCallback use: raw execute_cmd + _wrap_obs_info."""
        obs, _r, _d, info = self.sim.env.execute_cmd(cmd)
        obs, info = self.sim._wrap_obs_info(obs, info)
        self._winfo = info
        if isinstance(obs, dict) and obs.get("image") is not None:
            self.frame = obs["image"]
        return info

    @staticmethod
    def _inv_from_info(info):
        """Aggregate the 36-slot item inventory into {type: total_count}."""
        inv = {}
        d = (info or {}).get("inventory") or {}
        for item in (d.values() if isinstance(d, dict) else []):
            try:
                t = item.get("type"); q = int(item.get("quantity", 0))
            except Exception:
                continue
            # MineRL sometimes hands back the item type as a 0-d/1-element ndarray (or bytes)
            # instead of a str. `inv[t]` then raised "unhashable type: numpy.ndarray" INSIDE
            # step(), the slot's episode ended, and the client masked those 8 samples off for the
            # rest of the iteration -- 8 of 32 episodes scoring 0 with no error visible anywhere
            # except this server's own log.
            if isinstance(t, np.ndarray):
                t = t.item() if t.size == 1 else None
            if isinstance(t, (bytes, bytearray)):
                t = t.decode("utf-8", "replace")
            if t is not None and not isinstance(t, str):
                t = str(t)
            if t and t != "air" and q > 0:
                inv[t] = inv.get(t, 0) + q
        return inv

    def _inv_now(self):
        return self._inv_from_info(getattr(self, "_winfo", None) or {})

    def _noop_tick(self):
        """One no-op sim tick + wrapped-info refresh. Command effects (/give, /clear,
        /replaceitem) land on a LATER tick than the command itself; InitInventoryCallback
        settles the same way (steps noops until the inventory confirms)."""
        obs, _r, _d, info = self.sim.env.step(self.sim.env.noop_action())
        obs, info = self.sim._wrap_obs_info(obs, info)
        self._winfo = info
        if isinstance(obs, dict) and obs.get("image") is not None:
            self.frame = obs["image"]
        return info

    def _settle(self, pred, max_ticks=12):
        """Noop-tick until pred(inventory) is true or the budget runs out; returns final truth."""
        for _ in range(max_ticks):
            if pred(self._inv_now()):
                return True
            self._noop_tick()
        return pred(self._inv_now())

    @staticmethod
    def _inv_pick(inv, pat):
        if pat == "#log":    return [t for t in inv if t.endswith("_log")]
        if pat == "#planks": return [t for t in inv if t.endswith("_planks")]
        # smelting fuel: prefer coal (abundant underground, and burns 8 items in real MC)
        if pat == "#fuel":   return ([t for t in inv if t == "coal"] or
                                     [t for t in inv if t.endswith("_planks")] or
                                     [t for t in inv if t == "charcoal"])
        return [t for t in inv if t == pat]


    def _scripted_reward(self, gave):
        """Stage reward for a SCRIPTED craft.

        RewardsCallback only sees the sim's own statistics, and a scripted craft is executed with
        /give, which does not increment Minecraft's craft_item counter. Every craft_item stage in
        the reward config would be unreachable whenever MC_SCRIPTED_CRAFT=1 (craft_item:* never
        fires for /give crafts), leaving only the mine_block stages able to pay. Settle those
        stages here instead,
        sharing the callback's reward_memory so max_reward_times still caps them exactly once.
        """
        total = 0.0
        mem = getattr(self.reward_cb, "reward_memory", None)
        if mem is None or not self._rc:
            return 0.0
        used_table = False
        for item, n in gave.items():
            for cfg in self._rc:
                if cfg.get("event") != "craft_item":
                    continue
                objs = cfg.get("objects") or []
                if item not in list(objs):
                    continue
                ident = cfg.get("identity")
                cap = int(cfg.get("max_reward_times", 1))
                for _ in range(int(n)):
                    if mem.get(ident, 0) >= cap:
                        break
                    mem[ident] = mem.get(ident, 0) + 1
                    total += float(cfg.get("reward", 0.0))
                if item in ("wooden_pickaxe", "stone_pickaxe", "iron_pickaxe",
                            "diamond_pickaxe", "crafting_table"):
                    used_table = True
        # a 3x3 recipe implies the table was used; the s3b stage keys on the sim's
        # interact_with_crafting_table counter, which scripted crafting likewise never touches
        if used_table:
            for cfg in self._rc:
                if cfg.get("event") == "custom" and "interact_with_crafting_table" in list(cfg.get("objects") or []):
                    ident = cfg.get("identity")
                    if mem.get(ident, 0) < int(cfg.get("max_reward_times", 1)):
                        mem[ident] = mem.get(ident, 0) + 1
                        total += float(cfg.get("reward", 0.0))
        return total

    def _scripted_step(self, verb, item, instr):
        r_scripted = 0.0
        self._noop_tick()            # refresh: a prior scripted turn's commands may still be settling
        inv = self._inv_now()
        msg, delta, ok = "", {}, False
        if verb == "scatter":
            # bad-spawn escape hatch (labeled wall): random-teleport within 64 blocks so a slot
            # that spawned with no trees in reach gets a fresh view instead of wasting the episode
            self._exec_cmd("/spreadplayers ~ ~ 0 64 false @p")
            for _ in range(6):
                self._noop_tick()    # let the chunk render before the next STEVE-1 turn
            ok = True
            msg = "SCATTER OK: moved to a new spot"
            delta = {"scripted:scatter": 1}
        elif verb == "equip":
            if item == "pickaxe":
                item = next((p for p in ("diamond_pickaxe", "iron_pickaxe", "stone_pickaxe",
                                         "wooden_pickaxe") if inv.get(p)), "wooden_pickaxe")
            if not item or not inv.get(item):
                msg = f"EQUIP FAIL: no {item or 'such item'} in inventory"
            else:
                # net move: remove the inventory copy, then materialize it in the selected hotbar slot
                # /replaceitem OVERWRITES the mainhand slot -- rescue whatever is held first,
                # or equipping the iron pickaxe would destroy the stone pickaxe (dry-test find)
                hand0 = ((getattr(self, "_winfo", {}) or {}).get("equipped_items") or {}).get("mainhand")
                held0 = hand0.get("type") if isinstance(hand0, dict) else (hand0 if isinstance(hand0, str) else None)
                self._exec_cmd(f"/clear @p minecraft:{item} 1")
                self._exec_cmd(f"/replaceitem entity @p weapon.mainhand minecraft:{item} 1")
                if held0 and held0 not in ("air", item):
                    self._exec_cmd(f"/give @p minecraft:{held0} 1")
                def _held(_inv, _self=self, _item=item):
                    hand = ((getattr(_self, "_winfo", {}) or {}).get("equipped_items") or {}).get("mainhand")
                    held = hand.get("type") if isinstance(hand, dict) else (hand if isinstance(hand, str) else None)
                    return held == _item
                ok = self._settle(_held)
                msg = f"EQUIP {'OK' if ok else 'FAIL'}: wanted {item} in mainhand"
                if ok: delta = {f"scripted:equip:{item}": 1}
        elif item is None or item not in _RECIPES:
            msg = f"CRAFT FAIL: unknown recipe in '{instr}'"
        else:
            rec = _RECIPES[item]
            missing = [n for n in rec["needs"] if not inv.get(n)]
            plan = []
            for pat, cnt in rec["consume"]:
                got = 0
                for t in sorted(self._inv_pick(inv, pat), key=lambda t: -inv[t]):
                    take = min(cnt - got, inv[t])
                    if take > 0:
                        plan.append((t, take)); got += take
                    if got >= cnt: break
                if got < cnt:
                    missing.append(f"{pat.lstrip('#')} x{cnt - got}")
            if missing:
                msg = "CRAFT FAIL: missing " + ", ".join(missing)
            else:
                base = {t: self._inv_now().get(t, 0) for t, _ in rec["give"]}
                for t, cnt in plan:
                    self._exec_cmd(f"/clear @p minecraft:{t} {cnt}")
                for t, cnt in rec["give"]:
                    self._exec_cmd(f"/give @p minecraft:{t} {cnt}")
                ok = self._settle(lambda iv: all(iv.get(t, 0) >= base[t] + cnt for t, cnt in rec["give"]))
                msg = f"CRAFT {'OK' if ok else 'VERIFY FAIL'}: " + ", ".join(f"{t} x{cnt}" for t, cnt in rec["give"])
                if ok:
                    r_scripted = self._scripted_reward(dict(rec["give"]))
                if ok: delta = {f"scripted:craft:{t}": c for t, c in rec["give"]}
        self.cum += r_scripted
        print(f"[server] slot{self.idx} SCRIPTED {verb} {item}: {msg}"
              + (f"  (+{r_scripted:g})" if r_scripted else ""), flush=True)
        self._vid_frame(getattr(self, "_winfo", {}) or {}, f"[scripted] {instr}")
        self._log_turn(f"[scripted] {instr}", r_scripted, delta)
        done = self.n >= self.max_n
        if getattr(self, "_traj", None) is not None:
            f = _traj_save_frame(self.frame, f"{_RUN_TAG}_ep{self._ep_id:05d}_t{self.n:02d}.png")
            self._traj["turns"].append({"turn": self.n, "raw_action": instr[:800],
                                        "instruction": instr, "valid": True, "scripted": True,
                                        "result": msg, "reward": r_scripted, "frame": f, "delta": delta,
                                        "obs_text": self._obs_text_before,
                                        "events": {}})
            if done:
                self._traj["total_reward"] = self.cum
                self._traj["final_totals"] = self._totals()
                _traj_write(self._traj); self._traj = None
        if done: self._vid_close()
        return {"text": self._text(msg), "reward": r_scripted, "done": done,
                "image_b64": png_b64(self.frame),
                "info": {"data_source": self.ds, "won": self.cum > 0, "is_action_valid": ok,
                         "goal_reached": self._goal_reached(),
                         "instruction": instr, "scripted": True, "result": msg, "delta": delta,
                         "inventory": self._inv_now(), "events": {}}}

    _DESK_OF = {}
    for _d, _vs, _c in (("LOGGER", ("chop","get","gather","break","find","scatter"), (110,210,120)),
                        ("MINER",  ("dig","mine","explore","tunnel"),                (210,170,90)),
                        ("SMITH",  ("craft","equip","smelt"),                        (110,140,240))):
        for _v in _vs:
            _DESK_OF[_v] = (_d, _c)

    @staticmethod
    def _desk_of(instr):
        w = (instr or "").strip().lower().split()
        return Slot._DESK_OF.get(w[0], ("", (200, 200, 200))) if w else ("", (200, 200, 200))

    # ---- per-episode video (see VIDEO_DIR above) -----------------------------------------------
    _VID_BAR = 78

    def _vid_start(self, info):
        self._vw = None
        if not VIDEO_DIR or self.idx >= VIDEO_SLOTS:
            return
        try:
            import cv2
            pov = info.get("pov")
            if pov is None:
                return
            h, w = np.asarray(pov).shape[:2]
            self._vid_ep = getattr(self, "_vid_ep", 0) + 1
            self._vid_tmp = os.path.join(VIDEO_DIR, f"tmp_slot{self.idx}_ep{self._vid_ep:03d}.mp4")
            self._vid_n = 0
            # mp4v (MPEG-4 Part 2) is inter-predicted with no periodic intra refresh: once
            # one frame's bitstream corrupts, every frame depending on it collapses, worse the
            # further out -- measured on one 7-minute video: t=60s fine, t=180s partly
            # broken, t=400s whole frames stretched into one-pixel-row vertical stripes,
            # 11432 decode errors across the file; the same code produced zero errors under
            # lighter concurrency. Higher-scoring videos break worse
            # (longer play, more accumulation) -- exactly opposite to the "want long videos"
            # requirement. avc1 (H.264) has periodic IDR so one broken frame cannot poison
            # the segment, and browsers only accept it anyway.
            # H.264 is unavailable here: this environment's OpenCV was built without
            # libx264 ("Could not find encoder for codec_id=27"), so neither avc1 nor H264
            # opens a writer -- only mp4v works. So mp4v stays, and MC_VIDEO_FRAMES=1 offers
            # the reliable alternative: store a PNG frame sequence and assemble with ffmpeg
            # afterwards (the offline visualisation tool has always done this; its output has never broken).
            # mp4v's failure mode is described in the note below.
            _fourcc = cv2.VideoWriter_fourcc(*os.environ.get("MC_VIDEO_FOURCC", "mp4v"))
            self._vw = cv2.VideoWriter(self._vid_tmp, _fourcc,
                                       VIDEO_FPS, (w, h + self._VID_BAR))
            self._vid_wh = (w, h)
            self._vid_frame(info, "(episode start)")
        except Exception as e:
            print(f"[server] slot{self.idx} video start failed: {e}", flush=True)
            self._vw = None

    def _vid_frame(self, info, instr):
        if getattr(self, "_vw", None) is None:
            return
        try:
            import cv2
            pov = info.get("pov")
            if pov is None:
                return
            w, h = self._vid_wh
            fr = cv2.cvtColor(np.asarray(pov, dtype=np.uint8), cv2.COLOR_RGB2BGR)
            desk, dcol = self._desk_of(instr) if DESK_LABELS else ("", (90, 210, 255))
            bar = np.full((self._VID_BAR, w, 3), 18, dtype=np.uint8)
            if desk:
                cv2.rectangle(bar, (0, 0), (w, 4), dcol, -1)      # author stripe
            cv2.putText(bar, f"turn {self.n}/{self.max_n}  cum reward {self.cum:g}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (110, 240, 130) if self.cum > 0 else (200, 200, 200), 1, cv2.LINE_AA)
            if desk:
                cv2.putText(bar, desk, (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            dcol, 2, cv2.LINE_AA)
            cv2.putText(bar, f'"{(instr or "")[:88]}"', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, dcol if desk else (90, 210, 255),
                        1, cv2.LINE_AA)
            _composed = np.vstack([fr, bar])
            self._vw.write(_composed)
            # Optional: also store a PNG frame sequence. mp4v is inter-predicted with no
            # periodic intra refresh, so one corrupted frame takes the rest of the segment
            # down with it (measured: one 7-minute video fine at t=60s, whole
            # frames degraded to one-pixel-row vertical stripes by t=400s, 11432 decode
            # errors; higher-scoring videos break worse because they run longer). This
            # environment's OpenCV lacks libx264, so avc1/H264 cannot open a writer and only
            # mp4v works. PNG frames + ffmpeg afterwards (the binary DOES ship libx264) is
            # the only path verified reliable on long videos -- the offline visualisation tool has always
            # done it and its output has never broken. The cost is disk, hence default off.
            if FRAME_DIR:
                d = os.path.join(FRAME_DIR, f"slot{self.idx}_ep{self._vid_ep:03d}")
                os.makedirs(d, exist_ok=True)
                cv2.imwrite(os.path.join(d, f"f{self._vid_n:06d}.png"), _composed,
                            [cv2.IMWRITE_PNG_COMPRESSION, 3])
                self._vid_n += 1
        except Exception:
            self._vw = None   # one bad frame disables recording for this episode, never the rollout

    def _vid_close(self):
        vw = getattr(self, "_vw", None)
        if vw is None:
            return
        self._vw = None
        try:
            import os as _os
            vw.release()
            final = os.path.join(VIDEO_DIR, f"slot{self.idx}_ep{self._vid_ep:03d}_r{int(self.cum)}.mp4")
            _os.rename(self._vid_tmp, final)
            print(f"[server] slot{self.idx} video -> {final}", flush=True)
        except Exception as e:
            print(f"[server] slot{self.idx} video close failed: {e}", flush=True)

    # ---- rolling history -----------------------------------------------------------------------
    # Without history the prompt length is identical on every turn, i.e. the agent re-derives its
    # plan from a single frame each turn with no memory of what it has already done. That makes a
    # MULTI-STAGE task unsolvable: you cannot know whether to chop, craft or place without knowing
    # what you already have.
    #
    # OFF BY DEFAULT (MC_HISTORY=0) at the server level; scripts/start_mc_env.sh sets the history
    # length the MPT agents were trained with. MC_HISTORY=K shows the last K turns.
    HISTORY_K = int(os.environ.get("MC_HISTORY", "0"))

    # info categories worth diffing into a per-turn "what changed" line. mine_block/craft_item are
    # the tech-tree signal; custom carries interact_with_crafting_table etc. (74 counters, verified
    # against mc_constants.1.16.json).
    _EVENT_CATS = ("mine_block", "craft_item", "custom", "pickup")

    @staticmethod
    def _snap(info, cats=_EVENT_CATS):
        out = {}
        if isinstance(info, dict):
            for c in cats:
                d = info.get(c)
                if isinstance(d, dict):
                    for k, v in d.items():
                        try:
                            n = int(v.item() if hasattr(v, "item") else v)
                        except Exception:
                            continue
                        if n:
                            out[f"{c}:{k}"] = n
        return out

    def _delta(self, info):
        """Counters that went UP this turn, as {'craft_item:oak_planks': +4}."""
        cur = self._snap(info)
        prev = getattr(self, "_prev_snap", {}) or {}
        self._prev_snap = cur
        return {k: cur[k] - prev.get(k, 0) for k in cur if cur[k] > prev.get(k, 0)}

    def _goal_reached(self) -> bool:
        """True once the task's final stage (the last reward_cfg entry; for the iron task the
        iron_pickaxe craft) has been rewarded in this episode. Read from the reward callback's
        per-stage memory, which also caps max_reward_times."""
        try:
            rc = list(self._rc or [])
            if not rc:
                return False
            mem = getattr(getattr(self, "reward_cb", None), "reward_memory", None) or {}
            return int(mem.get(str(dict(rc[-1]).get("identity", "")), 0) or 0) > 0
        except Exception:  # noqa: BLE001 - a metric must never break a step
            return False

    def _totals(self):
        """Cumulative event totals = this sim's live counters + the branch correction.
        _snap_shift is {} everywhere except a /restore-d branch (where it holds
        parent-totals-at-branch minus this sim's counters at restore), so the non-branch
        path returns exactly what dict(_prev_snap) used to."""
        snap = getattr(self, "_prev_snap", {}) or {}
        shift = getattr(self, "_snap_shift", None)
        if not shift:
            return dict(snap)
        out = {}
        for k in set(snap) | set(shift):
            v = snap.get(k, 0) + shift.get(k, 0)
            if v > 0:
                out[k] = v
        return out

    @staticmethod
    def _got_str(delta):
        # delta carries MineRL's own stat counters (custom:play_one_minute, stat:sprint_one_cm, ...)
        # alongside the events that mean something. Taking the first four keys unfiltered let those
        # counters fill the line, so 74.6% of turns -- the ones that produced NOTHING -- still read
        # as "got time_since_rest x80, play_one_minute x80, ...". The agent could not tell a
        # productive turn from a wasted one anywhere in its history, which is exactly the signal a
        # relocate-when-stuck decision depends on. Real events survive the cut (only 0.7% of turns
        # had one crowded out), so this only removes noise.
        real = [(k, v) for k, v in delta.items() if k.split(":")[0] not in ("custom", "stat")]
        return ", ".join(f"{k.split(':')[-1]} x{v}" for k, v in real[:4]) or "nothing new"

    # Failure hint, ONLY on the current Last line -- history lines keep their plain shape (minus
    # the reward number). Pure function of (instr, delta, held), where held is the same Holding
    # state the observation prints.
    _HINT_VERBS = ("dig", "mine", "tunnel")

    @staticmethod
    def _hint_for(instr, delta, held):
        if any(k.split(":")[0] not in ("custom", "stat") for k in (delta or {})):
            return ""                                  # the turn produced something real
        w = (instr or "").strip().lower().split()
        if not w or w[0] not in Slot._HINT_VERBS:
            return ""
        held = held or ""
        if held == "wooden_pickaxe":
            return " (note: wooden pickaxe cannot harvest iron ore - see RULES)"
        if not held.endswith("_pickaxe"):
            return " (nothing drops: no pickaxe EQUIPPED - see RULES)"
        return ""                                      # stone or better: location problem, no guess

    def _hint(self, instr, delta):
        info = getattr(self, "_winfo", None) or {}
        hand = (info.get("equipped_items") or {}).get("mainhand") or {}
        hand = hand.get("type") if isinstance(hand, dict) else hand
        return self._hint_for(instr, delta, hand)

    def _log_turn(self, instr, reward, delta):
        if self.HISTORY_K <= 0:
            return
        # P1: the reward number is gone from the line ("-> reward 0; got" trained the policy on a
        # constant the evaluator renders nonzero); `reward` stays a parameter so callers are
        # untouched, it just no longer prints.
        self._hist.append(f'{self.n}. "{instr}" -> got {self._got_str(delta)}')

    def _history_block(self):
        if self.HISTORY_K <= 0 or not getattr(self, "_hist", None):
            return ""
        shown = self._hist[-self.HISTORY_K:]
        elided = len(self._hist) - len(shown)
        head = f"\n\nWhat you have already done ({'…%d earlier turns omitted; ' % elided if elided else ''}most recent last):\n"
        return head + "\n".join("  " + h for h in shown)

    def _inventory_block(self):
        """Cumulative totals, so the agent can see its own tech-tree state at a glance."""
        if self.HISTORY_K <= 0:
            return ""
        snap = self._totals()   # live counters + branch correction (identity when not branched)
        items = {k.split(":")[1]: v for k, v in snap.items() if k.startswith(("mine_block:", "craft_item:"))}
        return "\n\nTotals so far: " + (", ".join(f"{k} x{v}" for k, v in sorted(items.items())) or "nothing yet")


    _DIRS = ("south", "south-west", "west", "north-west", "north", "north-east", "east", "south-east")

    def _world_block(self):
        info = getattr(self, "_winfo", None) or {}
        loc = info.get("location_stats") or {}
        def g(k, d=None):
            v = loc.get(k, d)
            try: return float(v.item() if hasattr(v, "item") else v)
            except Exception: return d
        x, y, z = g("xpos"), g("ypos"), g("zpos")
        if x is None or y is None or z is None:
            return ""
        yaw = g("yaw", 0.0) or 0.0
        face = self._DIRS[int(((yaw % 360) + 22.5) // 45) % 8]
        if getattr(self, "_spawn_xz", None) is None:
            self._spawn_xz = (x, z)
        dx, dz = x - self._spawn_xz[0], z - self._spawn_xz[1]
        dist = (dx * dx + dz * dz) ** 0.5
        sky = g("can_see_sky", 1.0)
        light = g("light_level", 15.0)
        where = "underground" if (sky is not None and sky < 0.5) else "on the surface"
        hand = ((info.get("equipped_items") or {}).get("mainhand") or {})
        hand = hand.get("type") if isinstance(hand, dict) else hand
        life = info.get("life_stats") or {}
        def L(k):
            v = life.get(k)
            try: return int(float(v.item() if hasattr(v, "item") else v))
            except Exception: return None
        hp, food = L("life"), L("food")
        parts = [f"\n\nWorld: y={y:.0f} ({where}, light {light:.0f}/15), facing {face}, "
                 f"{dist:.0f} blocks from where you started"]
        parts.append(f"\nHolding: {hand or 'nothing'}")
        if hp is not None:
            parts.append(f"   health {hp}/20" + (f", food {food}/20" if food is not None else ""))
        return "".join(parts)

    def _text(self, last):
        """Observation text: TASK / RULES / Decision (no reward tail) / Totals /
        World+Holding / [WARNING] / history / Inventory / Craftable / Last / contract line.
        Every block is a deterministic function of already-rendered state (the format-SFT agents
        were trained on exactly these bytes)."""
        inv_line = ""
        if SCRIPTED:
            iv = self._inv_now()
            inv_line = ("\nInventory: " + (", ".join(f"{k} x{v}" for k, v in sorted(iv.items())) or "empty")
                        + "\n" + _craftable_line(iv, self.ladder))
        # WARNING: >=5 trailing shown-history lines with no real gain. Computed from the same
        # lines the observation prints (not a hidden counter), so it is recomputable from the
        # observation alone.
        warn = ""
        if self.HISTORY_K > 0 and getattr(self, "_hist", None):
            streak = 0
            for h in reversed(self._hist[-self.HISTORY_K:]):
                if not h.endswith("got nothing new"):
                    break
                streak += 1
            if streak >= 5:
                warn = "\n" + WARNING_TEXT
        return (f"TASK: {self.task}\n{self.rules}\nDecision {self.n}/{self.max_n}"
                f"{self._inventory_block()}{self._world_block()}{warn}{self._history_block()}{inv_line}"
                f"\nLast: {last}\n{CONTRACT_TEXT}")


def make_app(n_slots, K, obs_size, device, action_mode="steve1"):
    # MC_STEVE_SHARD=1: place slot i's STEVE inference on cuda:(i % n_gpu) instead of all
    # slots sharing cuda:0. env step is ~100% STEVE (measured), and the ~800 forwards/step
    # otherwise serialize on one GPU while the rest of the node idles. Per-device policies
    # are loaded lazily on first use (or preloaded at boot below).
    def _slot_device(i):
        if _traj_cfg("MC_STEVE_SHARD", ".mc_steve_shard", "0") != "1" or not str(device).startswith("cuda"):
            return device
        try:
            import torch
            n = torch.cuda.device_count()
        except Exception:
            n = 1
        return f"cuda:{i % max(n, 1)}"

    slots = [Slot(i, K, obs_size, _slot_device(i)) for i in range(n_slots)]
    print(f"[server] action_mode={action_mode} slot_cls={type(slots[0]).__name__} "
          f"steve_shard={os.environ.get('MC_STEVE_SHARD','0')} "
          f"devices={sorted({s.device for s in slots})}", flush=True)

    # --- slot retirement ------------------------------------------------------------------------
    # A JVM that fails to bind its Malmo port at launch ("java.net.BindException: Address already
    # in use") never answers: the slot's first reset blocks in a socket read past RESET_DEADLINE_S,
    # the orphaned thread keeps that slot's _op_lock, and every later reset queues behind it and
    # times out again. Fleet-wide that was 12-25% phantom episodes, +300s on every reset, and in
    # the end the whole box (the stale reset finally returned and raised inside the rebuild path).
    # So a slot whose reset/step raises or misses the deadline is RETIRED on the spot: a fresh Slot
    # object takes its index and builds a new JVM (new port) on its next reset; the old object is
    # dropped and its JVM closed in the background. Healthy slots are untouched. The client asks
    # for /reset again on the failed indices only (body "idx", see envs.py), so no episode is ever
    # served from a slot that did not come up.
    slot_cls = type(slots[0])
    _retire_lock = threading.Lock()
    _rebuilds = {}
    def _retire(i, why):
        with _retire_lock:
            old = slots[i]
            if getattr(old, "_retired", False):
                return
            old._retired = True
            slots[i] = slot_cls(i, K, obs_size, old.device)
            _rebuilds[i] = _rebuilds.get(i, 0) + 1
        _say(f"[server] slot{i} RETIRED ({why}) -> fresh slot, JVM rebuilt on its next reset "
             f"(rebuild #{_rebuilds[i]})")
        def _close():
            try:
                if getattr(old, "sim", None) is not None:
                    old.sim.close()
            except Exception as e:  # noqa: BLE001
                _say(f"[server] slot{i} old JVM close: {type(e).__name__}: {e}")
        threading.Thread(target=_close, daemon=True).start()

    async def health(req):
        # Full config fingerprint, not just liveness: every artifact that records /health then
        # carries the settings that produced it. --obs and cond_scale used to appear in no
        # artifact whatsoever, so a wrong value could not be detected afterwards.
        return web.json_response({"ok": True, "slots": n_slots, "frames_per_action": K,
                                  "steve_loaded": bool(_STEVE), "rebuilds": dict(_rebuilds),
                                  "obs": list(obs_size), "action_mode": action_mode,
                                  "cond_scale": getattr(slots[0], "cond_scale", None),
                                  "mc_history": int(os.environ.get("MC_HISTORY", "0")),
                                  "scripted_craft": bool(SCRIPTED),
                                  "mc_render": os.environ.get("MC_RENDER"),
                                  "mc_socktime": os.environ.get("MC_SOCKTIME"),
                                  "mc_snapshot": os.environ.get("MC_SNAPSHOT", "0"),
                                  "traj_dump": bool(TRAJ_DIR), "video_frames": bool(FRAME_DIR)})

    async def reset(req):
        # Slots reset CONCURRENTLY, exactly as /step and /restore already drive them: a hard
        # reset is ~28 s of JVM work, so the sequential loop this replaces cost 32 x 28 s = 15
        # min of every training iteration (~14% of a 108 min step) with the node's 128 cores
        # idle. Each slot holds its own _op_lock and its own Malmo instance, and asyncio.gather
        # preserves order, so out[i] still belongs to slot i.
        body = await req.json(); kws = body.get("slots", [])
        # "idx": reset only these slot indices (kws[k] belongs to idx[k]); absent -> slots 0..n-1.
        # Lets a client re-request just the slots whose reset failed instead of the whole box.
        idx = body.get("idx")
        if idx is None:
            kws = kws[:n_slots]; idx = list(range(len(kws)))
        else:
            idx = [int(i) for i in idx][:n_slots]; kws = kws[:len(idx)]
        def _one(i, kw):
            t0 = time.time()
            try:
                with slots[i]._op_lock:
                    r = slots[i].reset(kw)
                slots[i]._timed_out = False       # a successful reset revives a timed-out slot
            except Exception as e:
                traceback.print_exc()
                # never hand back a null frame: the trainer builds one prompt per slot and an
                # image-less prompt has no pixel_values, which kills the whole micro-batch
                _retire(i, f"reset raised {type(e).__name__}: {str(e)[:80]}")
                r = {"text": f"RESET ERROR: {e}", "image_b64": _last_frame_b64(slots[i]),
                     "info": {"data_source": "minecraft", "won": False, "error": str(e)}}
            print(f"[server] reset slot{i} {time.time()-t0:.1f}s", flush=True)
            return r
        t_all = time.time()

        async def _one_bounded(i, kw):
            """One slot's reset, with a deadline.

            asyncio.gather has no timeout, so a single slot that HANGS -- blocked on its
            _op_lock or inside a JVM socket read, neither of which raises -- makes /reset never
            return. The client then hits its own 3600s timeout and retries, which resets the
            other slots again and hangs on the same one: three transfer-eval runs burned ~6h
            that way, each with slot7 alone missing from every round. A slot that misses the
            deadline is reported like any other failed slot; the caller already pads and marks
            those terminal, so the episode set degrades from 8 usable slots to 7 instead of 0.
            The orphaned thread is left to finish on its own -- to_thread cannot be cancelled --
            and that slot is simply not trusted for this episode.
            """
            try:
                return await asyncio.wait_for(asyncio.to_thread(_one, i, kw), RESET_DEADLINE_S)
            except asyncio.TimeoutError:
                _retire(i, "reset exceeded deadline")
                slots[i]._timed_out = True        # the fresh slot refuses /step until a reset succeeds
                print(f"[server] reset slot{i} EXCEEDED {RESET_DEADLINE_S:.0f}s deadline -- "
                      f"reporting it failed so the other slots can proceed", flush=True)
                return {"text": f"RESET TIMEOUT after {RESET_DEADLINE_S:.0f}s",
                        "image_b64": _last_frame_b64(slots[i]),
                        "info": {"data_source": "minecraft", "won": False,
                                 "error": "reset timeout"}}

        out = list(await asyncio.gather(
            *[_one_bounded(i, kw) for i, kw in zip(idx, kws)]))
        print(f"[server] reset x{len(out)} {time.time()-t_all:.1f}s total", flush=True)
        return web.json_response({"obs": out})

    async def step(req):
        body = await req.json(); acts = body.get("actions", [])
        t0 = time.time()
        # Slots were stepped in a blocking for-loop, so one /step cost the SUM of 8 slots even
        # though the slots are independent JVMs. About half of each slot's time is spent waiting
        # on the JVM socket and the rest inside torch CUDA calls -- both release the GIL -- so a
        # thread per slot overlaps them. This is the single biggest lever on RL wall-clock: an RL
        # step is 120 sequential env turns, and nothing else shortens a turn.
        def _one(i, a):
            try:
                with slots[i]._op_lock:
                    return slots[i].step(a)
            except Exception as e:
                traceback.print_exc()
                if getattr(slots[i], "sim", None) is not None:      # a fresh, never-reset slot is not broken
                    _retire(i, f"step raised {type(e).__name__}: {str(e)[:80]}")
                return {"text": f"STEP ERROR: {e}", "reward": 0.0, "done": True,
                        "image_b64": _last_frame_b64(slots[i]),
                        "info": {"data_source": "minecraft", "won": False, "error": str(e)}}
        async def _step_bounded(i, a):
            if getattr(slots[i], "_timed_out", False):
                return {"text": "", "reward": 0.0, "done": True,
                        "info": {"data_source": "minecraft", "won": False,
                                 "error": "slot timed out earlier", "is_action_valid": False}}
            """One slot's step, with the same deadline /reset uses.

            asyncio.gather has no timeout, so a slot that hangs inside a JVM socket read makes
            /step never return either -- the transfer eval hung on its very first step for 43
            minutes with zero step events logged. A slot that misses the deadline is reported
            terminal; the caller pads short shards already, so the batch degrades by one slot
            instead of stalling the whole run.
            """
            try:
                return await asyncio.wait_for(asyncio.to_thread(_one, i, a), RESET_DEADLINE_S)
            except asyncio.TimeoutError:
                _retire(i, "step exceeded deadline")
                slots[i]._timed_out = True
                _say(f"[server] step slot{i} EXCEEDED {RESET_DEADLINE_S:.0f}s deadline -- "
                     f"marking it out until a reset revives it")
                return {"text": "", "reward": 0.0, "done": True,
                        "info": {"data_source": "minecraft", "won": False,
                                 "error": "step timeout", "is_action_valid": False}}

        out = list(await asyncio.gather(
            *[_step_bounded(i, a) for i, a in enumerate(acts[:n_slots])]))
        _r = [o.get("reward", 0.0) for o in out]
        _v = sum(1 for o in out if o.get("info", {}).get("is_action_valid"))
        _ins = [o.get("info", {}).get("instruction") for o in out if o.get("info", {}).get("instruction")]
        print(f"[server] step x{len(acts)} {time.time()-t0:.1f}s "
              f"reward_sum={sum(_r):g} reward_max={max(_r) if _r else 0:g} valid={_v}/{len(out)} "
              f"instr={_ins[:3]}", flush=True)
        return web.json_response({"obs": out})

    app = web.Application(client_max_size=64*1024*1024)

    async def step_slot(req):
        """{"slot": i, "action": "..."} -> {"obs": {...}} -- drive exactly ONE slot.

        The positional /step above advances EVERY slot it receives an action for, so a
        counterfactual branch could only be stepped by also stepping the whole mainline
        (padding "" still costs each slot a turn). This endpoint is the branch driver's
        step primitive: one slot moves, every other slot is untouched. The reply is the
        same per-slot obs dict /step puts into its list.
        """
        body = await req.json()
        i = int(body.get("slot", -1))
        if not (0 <= i < n_slots):
            return web.json_response({"error": f"slot {i} out of range 0..{n_slots-1}"}, status=400)
        a = body.get("action", "")
        def _one():
            try:
                with slots[i]._op_lock:
                    return slots[i].step(a)
            except Exception as e:
                traceback.print_exc()
                return {"text": f"STEP ERROR: {e}", "reward": 0.0, "done": True,
                        "image_b64": _last_frame_b64(slots[i]),
                        "info": {"data_source": "minecraft", "won": False, "error": str(e)}}
        out = await asyncio.to_thread(_one)
        return web.json_response({"obs": out})

    async def snapshot(req):
        """{"slots":[i,...]} -> {"snaps":[...]}; omitting slots snapshots every slot."""
        body = await req.json()
        idxs = body.get("slots") or list(range(n_slots))
        def _one(i):
            try:
                with slots[i]._op_lock:
                    return slots[i].snapshot()
            except Exception as e:
                traceback.print_exc(); return {"error": str(e)}
        out = list(await asyncio.gather(*[asyncio.to_thread(_one, i) for i in idxs[:n_slots]]))
        return web.json_response({"snaps": out})

    async def restore(req):
        """{"items":[{"slot":i,"snap":{...}},...]} -> {"results":[...]}.
        For branching: restore several slots to the SAME snapshot, then feed them different
        actions."""
        body = await req.json()
        items = body.get("items", [])
        out = []
        def _one(it):
            i = int(it["slot"])
            try:
                with slots[i]._op_lock:
                    return slots[i].restore(it["snap"], build=bool(it.get("build")))
            except Exception as e:
                traceback.print_exc(); return {"ok": False, "error": str(e)}
        out = list(await asyncio.gather(*[asyncio.to_thread(_one, it) for it in items[:n_slots]]))
        return web.json_response({"results": out})

    async def cmd(req):
        """{"slots":[i,...], "cmds":["/give @p ...", ...]} -> run commands on live sims.

        Why it exists: constructing a DETERMINISTIC mid-game state is a precondition of the
        counterfactual experiments (warming up via STEVE-1 tree-chopping is unreliable -- E6
        measured same-seed 8-slot runs where some slots got nothing in 25 steps). reset's
        inventory parameter requires InitInventoryCallback and only applies at sim build;
        this endpoint goes through Slot._exec_cmd on the existing execute_cmd channel and is
        immediately usable on a RUNNING slot, no rebuild.
        """
        body = await req.json()
        idxs = body.get("slots") or list(range(n_slots))
        cmds = body.get("cmds") or []
        out = []
        for i in idxs[:n_slots]:
            try:
                with slots[i]._op_lock:
                    for c in cmds:
                        slots[i]._exec_cmd(c)
                    out.append({"ok": True, "n": len(cmds),
                                "inv": slots[i]._inv_now() if SCRIPTED else {}})
            except Exception as e:
                traceback.print_exc(); out.append({"ok": False, "error": str(e)})
        return web.json_response({"results": out})

    async def _size_thread_pool(_app):
        # asyncio.to_thread's default executor caps at min(32, cpu_count + 4) workers, so above
        # 32 slots /reset, /step and /cmd would silently serialize into waves -- a throughput
        # cliff that reads as "the env got slower" rather than as a configuration limit. One
        # worker per slot plus headroom keeps every per-slot endpoint fully concurrent; the
        # threads sit blocked on a JVM socket, not on Python. Set from a startup hook because
        # web.run_app builds its OWN event loop, so sizing the pool before it starts is a no-op.
        import concurrent.futures as _cf
        n_workers = max(n_slots + 4, 32)
        asyncio.get_running_loop().set_default_executor(
            _cf.ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="slot"))
        print(f"[server] slot thread pool: {n_workers} workers", flush=True)

    app.on_startup.append(_size_thread_pool)
    app.add_routes([web.get("/health", health), web.post("/reset", reset), web.post("/step", step),
                    web.post("/step_slot", step_slot),
                    web.post("/snapshot", snapshot), web.post("/restore", restore),
                    web.post("/cmd", cmd)])
    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=31500)
    ap.add_argument("--slots", type=int, default=4)
    # NO DEFAULT IN steve1 MODE, on purpose. Canonical is 80 (all 69 August server logs
    # report K=80). The old default of 40 is what let a launcher silently run half the
    # ticks: action yield 15.7%->9.3%, the no-progress WARNING permanent, the policy
    # degenerating to teleport-spam, one week of results invalidated -- with every log
    # looking normal.
    ap.add_argument("--frames-per-action", type=int, default=None)
    ap.add_argument("--obs", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--action-mode", choices=["steve1"], default="steve1",
                    help="steve1: <action>text</action> -> STEVE-1 x K frames.")
    a = ap.parse_args()
    if a.frames_per_action is None:
        ap.error("--frames-per-action is REQUIRED in steve1 mode; the benchmark value is 80. "
                 "Passing nothing used to mean 40 and silently halved every action's effect.")
    _traj_init()
    load_steve(a.device)   # fail fast at boot rather than on the first step
    # preload STEVE on every GPU when sharding, so no slot pays a load stall mid-rollout
    if os.environ.get("MC_STEVE_SHARD", "0") == "1" and str(a.device).startswith("cuda"):
        import torch
        for _g in range(torch.cuda.device_count()):
            load_steve(f"cuda:{_g}")
    print(f"[server] listening on :{a.port} slots={a.slots} K={a.frames_per_action} "
          f"mode={a.action_mode} obs={a.obs} history={os.environ.get('MC_HISTORY','0')} "
          f"scripted_craft={os.environ.get('MC_SCRIPTED_CRAFT','0')} "
          f"render={os.environ.get('MC_RENDER','?')} socktime={os.environ.get('MC_SOCKTIME','?')} "
          f"snapshot={os.environ.get('MC_SNAPSHOT','0')}", flush=True)
    web.run_app(make_app(a.slots, a.frames_per_action, (a.obs, a.obs), a.device,
                         action_mode=a.action_mode), port=a.port, print=None)
