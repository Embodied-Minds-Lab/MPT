# Minecraft environment for the MPT trainer — CLIENT side.
#
# The trainer runs on python3.12 (torch/sglang); MineStudio needs python3.10 (JDK8/MineRL). They cannot
# share a process, so the heavy side (MinecraftSim slots + the STEVE-1 controller) lives in
# src/server/mc_env/mc_env_server.py and we talk to it over JSON/HTTP. Nothing here imports minestudio.
#
#   MPT VLM agent --(<action>text instruction</action>)--> [server] STEVE-1 -> K game frames
#                   <--(new screenshot + status text + event reward)---------
import base64
import io
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import requests

MC_ENV_URL = os.environ.get("MC_ENV_URL", "http://127.0.0.1:31500")
_TIMEOUT = float(os.environ.get("MC_ENV_TIMEOUT", "1200"))

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_ACTION = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)


def extract_instruction(text: str, max_words: int = 20) -> Optional[str]:
    """Agent must emit <action>...</action>; None means an invalid action."""
    vis = _THINK.sub("", text or "")
    m = _ACTION.search(vis)
    if not m:
        return None
    s = re.sub(r"\s+", " ", m.group(1)).strip()
    return " ".join(s.split()[:max_words]) if s else None


def _decode_image(b64: Optional[str]):
    if not b64:
        return None
    try:
        from PIL import Image
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    except Exception:
        return None


class MinecraftMultiProcessEnv:
    """MPT env contract, backed by the remote Minecraft slot pool."""

    def __init__(self, seed: int = 0, env_num: int = 1, group_n: int = 1, is_train: bool = True,
                 url: str = None):
        self.env_num = env_num
        self.group_n = group_n
        self.batch_size = env_num * group_n
        self.is_train = is_train
        self.seed = seed
        # MC_ENV_URL may name several servers, comma separated. Episodes are then split across
        # them in contiguous blocks sized by each server's own slot count, and every /reset and
        # /step fans out concurrently, so eight slots on four machines behave like one pool of
        # thirty-two -- without asking a single node to hold thirty-two Minecraft JVMs, a size
        # nothing here has ever run.
        raw = url or MC_ENV_URL
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
                    f"Minecraft env server unreachable at {u} ({e}). "
                    f"Start it with scripts/start_mc_env.sh and set MC_ENV_URL.") from e
            n = int(h.get("slots", 0))
            self.slots.append(n)
            self._health.append(h)
            total += n
            print(f"[minecraft] server OK at {u}: {h}", flush=True)
        if total < self.batch_size:
            raise RuntimeError(
                f"{len(self.urls)} server(s) offer {total} slots but this run needs "
                f"batch_size={self.batch_size}; add servers, raise --slots, or lower the batch")
        if total > self.batch_size:
            print(f"[minecraft] {total} slots available, using {self.batch_size}", flush=True)

    def _fanout(self, path: str, payloads: List[Dict]) -> List[Dict]:
        """POST one shard per server in parallel; reassemble the replies in batch order.

        A shard that comes back short (a server that lost slots) is padded with terminal blanks
        rather than silently shifting every later episode onto the wrong slot."""
        def one(i):
            if self.spans[i].stop <= self.spans[i].start:
                return []
            r = requests.post(f"{self.urls[i]}{path}", json=payloads[i], timeout=_TIMEOUT)
            if r.status_code != 200:
                # A bare r.json() on aiohttp's plain-text error page raises
                #   JSONDecodeError: Extra data: line 1 column 5 (char 4)
                # -- it parses "500" as a number and chokes on the "I" of "Internal". That message
                # names no server, no path and no cause, and it killed three multi-hour arms today
                # before anyone could tell it was even an HTTP error. Say what actually happened.
                raise RuntimeError(
                    f"env server {self.urls[i]} returned HTTP {r.status_code} on {path}: "
                    f"{r.text[:400]!r}")
            return r.json().get("obs", [])

        out: List[Dict] = []
        for i, items in enumerate(self._pool.map(one, range(len(self.urls)))):
            want = self.spans[i].stop - self.spans[i].start
            if len(items) < want:
                print(f"[minecraft] {self.urls[i]} returned {len(items)}/{want} slots on {path}",
                      flush=True)
                items = list(items) + [{"text": "", "reward": 0.0, "done": True, "info": {}}] * (
                    want - len(items))
            out.extend(items[:want])
        return out

    # A batch is trained only when EVERY slot came up (design rule: never start a step on a
    # partial batch; a slot that failed to reset is re-requested until it is good). The server retires
    # a failed slot and rebuilds its JVM on the next reset, so re-asking the same index is the fix, not
    # a retry of the same broken JVM. Each round is bounded by the server's own reset deadline (300s).
    RESET_ROUNDS = int(os.environ.get("MC_RESET_ROUNDS", "8"))

    @staticmethod
    def _reset_failed(item: Dict) -> bool:
        info = item.get("info", {}) or {}
        return bool(info.get("error")) or str(item.get("text", "")).startswith(("RESET TIMEOUT", "RESET ERROR"))

    def _post_some(self, path: str, payloads: List) -> List[List[Dict]]:
        """Like _fanout but only to the servers whose payload is not None; raw item lists back."""
        def one(i):
            if payloads[i] is None:
                return []
            r = requests.post(f"{self.urls[i]}{path}", json=payloads[i], timeout=_TIMEOUT)
            if r.status_code != 200:
                raise RuntimeError(f"env server {self.urls[i]} returned HTTP {r.status_code} on {path}: {r.text[:400]!r}")
            return r.json().get("obs", [])
        return list(self._pool.map(one, range(len(self.urls))))

    def reset(self, kwargs: List[Dict]):
        payload = []
        for i, kw in enumerate(kwargs[:self.batch_size]):
            kw = dict(kw)
            kw.setdefault("seed", self.seed + i)
            payload.append(kw)
        shards = [{"slots": payload[sp]} for sp in self.spans]
        items = self._fanout("/reset", shards)
        for rnd in range(1, self.RESET_ROUNDS + 1):
            bad = [j for j, it in enumerate(items) if self._reset_failed(it)]
            if not bad:
                break
            print(f"[minecraft] reset round {rnd}: {len(bad)} slot(s) failed {bad} "
                  f"({[str(items[j].get('text', ''))[:40] for j in bad[:3]]}) -> re-requesting those slots only",
                  flush=True)
            retry = []
            for i, sp in enumerate(self.spans):
                local = [j - sp.start for j in bad if sp.start <= j < sp.stop]
                retry.append({"slots": [payload[sp.start + l] for l in local], "idx": local} if local else None)
            replies = self._post_some("/reset", retry)
            for i, sp in enumerate(self.spans):
                if retry[i] is None:
                    continue
                for l, it in zip(retry[i]["idx"], replies[i]):
                    items[sp.start + l] = it
        else:
            bad = [j for j, it in enumerate(items) if self._reset_failed(it)]
            raise RuntimeError(f"Minecraft reset: {len(bad)} slot(s) still failing after {self.RESET_ROUNDS} rounds: "
                               f"{[(j, str(items[j].get('text', ''))[:60]) for j in bad]}")
        obs_list, info_list = [], []
        for item in items:
            obs_list.append(item.get("text", ""))
            info = dict(item.get("info", {}))
            info["image"] = _decode_image(item.get("image_b64"))
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
            info["image"] = _decode_image(item.get("image_b64"))
            info_l.append(info)
        return obs_l, rew_l, done_l, info_l

    # ---- counterfactual env adapter (the collector's cf_* contract; see
    # ---- multi_turn_rollout/counterfactual_collector.py) --------------------------------------
    # The mainline batch uses slots [0, batch_size) of ONE server; the two branch slots are the
    # next two, so the server must run MC_SNAPSHOT=1 (snapshot / restore / step_slot endpoints)
    # with --slots >= batch_size + 2.
    def cf_bind(self, timeout: float = None) -> List[int]:
        if timeout is not None:
            self._cf_timeout = float(timeout)
        if len(self.urls) != 1:
            raise RuntimeError(f"cf collector v1 supports exactly ONE MC server, got {self.urls} "
                               "(spare-slot spans across servers are not implemented)")
        total = int(self.slots[0])
        bs = int(self.batch_size)
        if total < bs + 2:
            raise RuntimeError(
                f"cf collector: server at {self.urls[0]} offers {total} slots but the run needs "
                f"batch_size+2 = {bs + 2} (two spare slots for branch rollouts). Relaunch the "
                f"server with --slots {bs + 2} and MC_SNAPSHOT=1.")
        h = self._health[0] if self._health else {}
        if "mc_snapshot" not in h:
            # a server older than the flag (same rule as scripts/train_mc_mpt.sh): a server
            # without MC_SNAPSHOT=1 answers /snapshot with {"error": ...}, which the collector
            # reports for every skipped branch point
            print(f"[minecraft] WARNING: /health at {self.urls[0]} does not report mc_snapshot; "
                  "the counterfactual bid phase needs the server launched with MC_SNAPSHOT=1",
                  flush=True)
        elif str(h.get("mc_snapshot")) != "1":
            raise RuntimeError(
                f"cf collector: server at {self.urls[0]} reports mc_snapshot={h.get('mc_snapshot')}; "
                "the counterfactual bid phase needs the server launched with MC_SNAPSHOT=1.")
        return [bs, bs + 1]

    def cf_snapshot(self, slot: int) -> Dict:
        r = requests.post(f"{self.urls[0]}/snapshot", json={"slots": [int(slot)]},
                          timeout=self._cf_timeout)
        snaps = r.json().get("snaps", [])
        return snaps[0] if snaps else {"error": "empty /snapshot reply"}

    def cf_restore(self, slot: int, snap: Dict) -> Dict:
        # build=True: spare slots start with NO sim (the mainline batch only resets slots
        # [0, TBS)), and successive bundles come from different mainline slots with different
        # world seeds -- the server (re)builds the spare's world from the snapshot on demand
        r = requests.post(f"{self.urls[0]}/restore",
                          json={"items": [{"slot": int(slot), "snap": snap, "build": True}]},
                          timeout=self._cf_timeout)
        res = r.json().get("results", [])
        return res[0] if res else {"ok": False, "error": "empty /restore reply"}

    def cf_step_slot(self, slot: int, action: str) -> Dict:
        r = requests.post(f"{self.urls[0]}/step_slot", json={"slot": int(slot), "action": action},
                          timeout=self._cf_timeout)
        body = r.json()
        if "obs" not in body:
            raise RuntimeError(f"/step_slot returned no obs: {str(body)[:300]}")
        obs = body["obs"]
        # /step_slot answers an exception with {"text": "STEP ERROR: ...", "info": {"error": ...}}
        _err = (obs.get("info") or {}).get("error")
        if _err or str(obs.get("text", "")).startswith("STEP ERROR"):
            raise RuntimeError(f"branch env step failed on slot {slot}: "
                               f"{_err or str(obs.get('text'))[:200]}")
        return obs

    def close(self):
        self._pool.shutdown(wait=False)


def build_minecraft_envs(seed: int = 0, env_num: int = 1, group_n: int = 1, is_train: bool = True,
                         frames_per_action: int = None, obs_size=(128, 128)):
    # frames_per_action / obs_size are properties of the SERVER (set at its launch); kept in the
    # signature so make_envs stays uniform with the other env packages.
    return MinecraftMultiProcessEnv(seed=seed, env_num=env_num, group_n=group_n, is_train=is_train)
