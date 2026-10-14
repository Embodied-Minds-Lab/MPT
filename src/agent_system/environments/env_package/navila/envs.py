# NaVILA robot-navigation environment for the MPT trainer -- CLIENT side.
#
# Habitat-sim 0.1.7 needs its own conda env (py3.10, numpy 1.26) and a GPU it can render on; the
# MPT training/rollout side runs py3.12 + sglang. They cannot share a process, so the heavy side (habitat slot pool + R2R-CE episodes + the mid-level-action parser
# and the official WaypointReward) lives in src/server/navila_env/navila_env_server.py and we talk to it over
# JSON/HTTP -- the exact structure MC uses. Nothing here imports habitat.
#
#   sglang NaVILA policy --(mid-level nav action text)--> [server] parse -> atomic habitat steps
#                        <--(8-frame RGB window + status text + waypoint reward + measures)------
import base64
import io
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import requests

NAVILA_ENV_URL = os.environ.get("NAVILA_ENV_URL", "http://127.0.0.1:31600")
_TIMEOUT = float(os.environ.get("NAVILA_ENV_TIMEOUT", "1200"))

# The policy emits e.g. "The next action is turn right 45 degree." -- the same phrasing the
# habitat server parses. The model emits <think>...</think>\n<act>ACTION</act>; a response is a
# valid action iff its <act> span names one of the four verbs.
_ACT_TAG = re.compile(r"<act>(.*?)</act>", re.IGNORECASE | re.DOTALL)
_ACTION_PATTERNS = [
    re.compile(r"\bstop\b", re.IGNORECASE),
    re.compile(r"\bmove forward\b", re.IGNORECASE),
    re.compile(r"\bturn left\b", re.IGNORECASE),
    re.compile(r"\bturn right\b", re.IGNORECASE),
]


def is_valid_action(text: str) -> bool:
    """True iff the <act>...</act> span names a recognizable navigation verb (server parses the rest)."""
    m = _ACT_TAG.search(text or "")
    if not m:
        return False
    span = m.group(1)
    return any(p.search(span) for p in _ACTION_PATTERNS)


def _decode_frames(frames_b64: Optional[List[str]]) -> Optional[List]:
    """Decode the 8-frame observation window (list of base64 PNGs) into a list of PIL RGB images."""
    if not frames_b64:
        return None
    try:
        from PIL import Image
        out = []
        for b in frames_b64:
            out.append(Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB"))
        return out
    except Exception:
        return None


class NavilaMultiProcessEnv:
    """MPT env contract, backed by the remote habitat slot pool. Mirrors MinecraftMultiProcessEnv,
    but each observation carries the full 8-frame video window (info["frames"]) that NaVILA sees."""

    def __init__(self, seed: int = 0, env_num: int = 1, group_n: int = 1, is_train: bool = True,
                 url: str = None):
        self.env_num = env_num
        self.group_n = group_n
        self.batch_size = env_num * group_n
        self.is_train = is_train
        self.seed = seed
        # NAVILA_ENV_URL may name several servers (comma separated); episodes split across them in
        # contiguous blocks sized by each server's slot count, every /reset and /step fanning out
        # concurrently -- so N slots on M machines behave like one pool of N*M.
        raw = url or NAVILA_ENV_URL
        self.urls = [u.strip().rstrip("/") for u in str(raw).split(",") if u.strip()]
        self.slots: List[int] = []
        self._health: List[Dict] = []     # each server's /health reply at construction
        self._cf_timeout = _TIMEOUT       # per-call timeout of the counterfactual endpoints
        self._check()
        self.spans: List[slice] = []
        lo = 0
        for cap in self.slots:
            if lo >= self.batch_size:
                self.spans.append(slice(lo, lo))
                continue
            hi = min(lo + cap, self.batch_size)
            self.spans.append(slice(lo, hi))
            lo = hi
        self._pool = ThreadPoolExecutor(max_workers=max(1, len(self.urls)))

    def _check(self):
        total = 0
        for u in self.urls:
            try:
                h = requests.get(f"{u}/health", timeout=30).json()
            except Exception as e:
                raise RuntimeError(
                    f"NaVILA env server unreachable at {u} ({e}). "
                    f"Start it with scripts/start_navila_env.sh and set NAVILA_ENV_URL.") from e
            n = int(h.get("slots", 0))
            self.slots.append(n)
            self._health.append(h)
            total += n
            print(f"[navila] server OK at {u}: {h}", flush=True)
        if total < self.batch_size:
            raise RuntimeError(
                f"{len(self.urls)} server(s) offer {total} slots but this run needs "
                f"batch_size={self.batch_size}; add servers, raise --slots, or lower the batch")
        if total > self.batch_size:
            print(f"[navila] {total} slots available, using {self.batch_size}", flush=True)

    def _fanout(self, path: str, payloads: List[Dict]) -> List[Dict]:
        """POST one shard per server in parallel; reassemble replies in batch order, padding a
        short shard with terminal blanks rather than shifting later episodes onto wrong slots."""
        def one(i):
            if self.spans[i].stop <= self.spans[i].start:
                return []
            r = requests.post(f"{self.urls[i]}{path}", json=payloads[i], timeout=_TIMEOUT)
            return r.json().get("obs", [])

        out: List[Dict] = []
        for i, items in enumerate(self._pool.map(one, range(len(self.urls)))):
            want = self.spans[i].stop - self.spans[i].start
            if len(items) < want:
                print(f"[navila] {self.urls[i]} returned {len(items)}/{want} slots on {path}",
                      flush=True)
                items = list(items) + [{"text": "", "reward": 0.0, "done": True, "info": {}}] * (
                    want - len(items))
            out.extend(items[:want])
        return out

    def reset(self, kwargs: List[Dict]):
        payload = []
        for i, kw in enumerate(kwargs[:self.batch_size]):
            kw = dict(kw)
            kw.setdefault("seed", self.seed + i)
            payload.append(kw)
        shards = [{"slots": payload[sp]} for sp in self.spans]
        obs_list, info_list = [], []
        for item in self._fanout("/reset", shards):
            obs_list.append(item.get("text", ""))
            info = dict(item.get("info", {}))
            info["frames"] = _decode_frames(item.get("frames_b64"))
            info_list.append(info)
        return obs_list, info_list

    def step(self, actions: List[str]):
        acts = list(actions)[:self.batch_size]
        shards = [{"actions": acts[sp]} for sp in self.spans]
        obs_l, rew_l, done_l, info_l = [], [], [], []
        for item in self._fanout("/step", shards):
            obs_l.append(item.get("text", ""))
            rew_l.append(float(item.get("reward", 0.0)))
            done_l.append(bool(item.get("done", True)))
            info = dict(item.get("info", {}))
            info["frames"] = _decode_frames(item.get("frames_b64"))
            info_l.append(info)
        return obs_l, rew_l, done_l, info_l

    # ---- counterfactual env adapter (the collector's cf_* contract; see
    # ---- multi_turn_rollout/counterfactual_collector.py) --------------------------------------
    # The mainline batch uses slots [0, batch_size) of ONE server; the two branch slots are the
    # next two, so the server must run NAVILA_SNAPSHOT=1 (snapshot / restore / step_slot
    # endpoints) with --slots >= batch_size + 2. The habitat server uses single-slot shapes:
    # /snapshot {"slot"} -> {"state"}, /restore {"slot", "state"} -> {"obs"}.
    def cf_bind(self, timeout: float = None) -> List[int]:
        if timeout is not None:
            self._cf_timeout = float(timeout)
        if len(self.urls) != 1:
            raise RuntimeError(f"cf collector v1 supports exactly ONE NaVILA server, got {self.urls} "
                               "(spare-slot spans across servers are not implemented)")
        total = int(self.slots[0])
        bs = int(self.batch_size)
        if total < bs + 2:
            raise RuntimeError(
                f"cf collector: server at {self.urls[0]} offers {total} slots but the run needs "
                f"batch_size+2 = {bs + 2} (two spare slots for branch rollouts). Relaunch the "
                f"server with --slots {bs + 2} and NAVILA_SNAPSHOT=1.")
        h = self._health[0] if self._health else {}
        if "navila_snapshot" not in h:
            # a server older than the flag: the endpoints answer {"error": "snapshot disabled"}
            # when it runs without NAVILA_SNAPSHOT=1, which fails every branch point loudly
            print(f"[navila] WARNING: /health at {self.urls[0]} does not report navila_snapshot; "
                  "the counterfactual bid phase needs the server launched with NAVILA_SNAPSHOT=1",
                  flush=True)
        elif str(h.get("navila_snapshot")) != "1":
            raise RuntimeError(
                f"cf collector: server at {self.urls[0]} reports navila_snapshot="
                f"{h.get('navila_snapshot')}; the counterfactual bid phase needs the server "
                "launched with NAVILA_SNAPSHOT=1.")
        return [bs, bs + 1]

    def cf_snapshot(self, slot: int) -> Dict:
        r = requests.post(f"{self.urls[0]}/snapshot", json={"slot": int(slot)}, timeout=self._cf_timeout)
        b = r.json()
        return b.get("state") or b

    def cf_restore(self, slot: int, snap: Dict) -> Dict:
        r = requests.post(f"{self.urls[0]}/restore", json={"slot": int(slot), "state": snap},
                          timeout=self._cf_timeout)
        b = r.json()
        return b.get("obs") or b

    def cf_step_slot(self, slot: int, action: str) -> Dict:
        r = requests.post(f"{self.urls[0]}/step_slot", json={"slot": int(slot), "action": action},
                          timeout=self._cf_timeout)
        body = r.json()
        if "obs" not in body:
            raise RuntimeError(f"/step_slot returned no obs: {str(body)[:300]}")
        obs = body["obs"]
        # the habitat server answers a slot exception with {"text": "SLOT ERROR: ...", "done": True}
        _err = (obs.get("info") or {}).get("error") if isinstance(obs, dict) else "non-dict obs"
        _text = str(obs.get("text", "")) if isinstance(obs, dict) else ""
        if _err or _text.startswith(("STEP ERROR", "SLOT ERROR")):
            raise RuntimeError(f"branch env step failed on slot {slot}: {_err or _text[:200]}")
        return obs

    def close(self):
        self._pool.shutdown(wait=False)


def build_navila_envs(seed: int = 0, env_num: int = 1, group_n: int = 1, is_train: bool = True):
    # Validation runs on a SEPARATE env server loaded with split=val_unseen. episode_ids collide
    # across splits (train and val_unseen both number from 1), and the server matches a reset by
    # episode_id alone -- so a val episode routed to the split=train server silently resolves to the
    # WRONG same-id train episode (and that server can't even load val scenes). NAVILA_ENV_URL_VAL
    # points at the val_unseen server; if unset we fall back to the train URL (legacy single-split).
    url = None
    if not is_train:
        url = os.environ.get("NAVILA_ENV_URL_VAL") or None
    return NavilaMultiProcessEnv(seed=seed, env_num=env_num, group_n=group_n, is_train=is_train, url=url)
