#!/usr/bin/env python
import argparse, asyncio, base64, collections, concurrent.futures, os, sys, time, traceback, uuid

import numpy as np
from aiohttp import web


class SteveService:
    def __init__(self, device="cuda", batch_wait_ms=4.0, max_batch=64, state_ttl_s=900.0):
        import torch
        from minestudio.models import SteveOnePolicy
        self.torch = torch
        repo = os.environ.get("STEVE1_CKPT", "CraftJarvis/MineStudio_STEVE-1.official")
        self.model = SteveOnePolicy.from_pretrained(repo).to(device).eval()
        self.device = device
        self.epoch = uuid.uuid4().hex[:8]     # changes on every restart: a client can tell
        self.batch_wait = batch_wait_ms / 1000.0
        self.max_batch = max_batch
        self.state_ttl = state_ttl_s
        # sid -> {"state": list[tensor] (batch dim 1, cloned), "req": last req id,
        #         "action": last action, "t": last used}
        self.states = {}
        self.conds = collections.OrderedDict()   # nonce -> mineclip_embeds [1, D]
        self.queue = collections.deque()         # pending (req dict, future)
        self.wake = asyncio.Event()
        # max_workers MUST stay 1: MaskedAttention.forward stores the attention mask on the
        # shared module before running the block, so two concurrent forwards on one policy
        # corrupt each other silently (same shapes, different contents).
        self.gpu = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="steve-gpu")
        self.stats = dict(frames=0, batches=0, batch_hist=collections.Counter(), errors=0, t_gpu=0.0,
                          state_miss=0, dup_req=0, last_batch_t=0.0)
        self.batcher_task = None
        print(f"[steve] loaded {repo} on {device}, epoch {self.epoch}", flush=True)

    # ---- conditioning ------------------------------------------------------------------------
    def _cond(self, r):
        key = r["nonce"]
        hit = self.conds.get(key)
        if hit is None:
            c = self.model.prepare_condition({"text": r["text"], "cond_scale": float(r["cond_scale"])},
                                             deterministic=bool(r.get("det", False)))
            hit = c["mineclip_embeds"]
            self.conds[key] = hit
            while len(self.conds) > 16384:
                self.conds.popitem(last=False)
        return hit

    # ---- one batched forward (runs in the single GPU thread) ---------------------------------
    def _run_batch(self, items):
        """items: list of request dicts (same cond_scale, det and image shape). Returns one
        result dict per item: {"action": ..., "stale": bool}."""
        torch = self.torch
        t0 = time.time()
        cs = float(items[0]["cond_scale"]); det = bool(items[0].get("det", False))
        results = [None] * len(items)
        live = []                                # (index, request) that really need a forward
        for i, r in enumerate(items):
            ent = self.states.get(r["sid"])
            if ent is not None and ent.get("req") == r.get("req"):
                results[i] = {"action": ent["action"], "stale": False, "dup": True}   # retry of a served frame
                self.stats["dup_req"] += 1
                continue
            live.append((i, r))
        if live:
            with torch.no_grad():
                imgs = torch.from_numpy(np.stack([r["img"] for _, r in live]))[:, None].to(self.device)  # [B,1,H,W,3]
                embeds = torch.cat([self._cond(r) for _, r in live], dim=0)                             # [B, D]
                cond = {"cond_scale": cs, "mineclip_embeds": embeds}
                per, stale = [], []
                for _, r in live:
                    ent = None if r.get("fresh") else self.states.get(r["sid"])
                    if ent is None and not r.get("fresh"):
                        self.stats["state_miss"] += 1
                        stale.append(True)
                    else:
                        stale.append(False)
                    per.append(ent["state"] if ent is not None else self.model.initial_state(cond, 1))
                state_in = [torch.cat([s[j] for s in per], dim=0) for j in range(len(per[0]))]
                act, state_out = self.model.get_action({"image": imgs, "condition": cond},
                                                       state_in=state_in, deterministic=det)
                now = time.time()
                for k, (i, r) in enumerate(live):
                    a = {kk: v[k].detach().cpu().numpy().squeeze().tolist() for kk, v in act.items()}
                    # clone: a slice of the batched state would pin the whole batch's tensors
                    self.states[r["sid"]] = {"state": [x[k:k + 1].clone() for x in state_out],
                                             "req": r.get("req"), "action": a, "t": now}
                    results[i] = {"action": a, "stale": stale[k], "dup": False}
        self.stats["t_gpu"] += time.time() - t0
        self.stats["last_batch_t"] = time.time()
        return results

    async def batcher(self):
        loop = asyncio.get_running_loop()
        while True:
            try:
                if not self.queue:
                    self.wake.clear()
                    await self.wake.wait()
                if len(self.queue) < self.max_batch:
                    await asyncio.sleep(self.batch_wait)        # let the other slots' frames arrive
                first = self.queue[0][0]
                key = self._key(first)
                items, futs, rest, seen = [], [], collections.deque(), set()
                while self.queue and len(items) < self.max_batch:
                    r, f = self.queue.popleft()
                    if f.done():                             # client went away
                        continue
                    # one forward per chain: a duplicate sid waits for the next batch
                    if self._key(r) == key and r["sid"] not in seen:
                        items.append(r); futs.append(f); seen.add(r["sid"])
                    else:
                        rest.append((r, f))
                self.queue.extendleft(reversed(rest))
                if not items:
                    continue
                try:
                    out = await loop.run_in_executor(self.gpu, self._run_batch, items)
                    for f, o in zip(futs, out):
                        if not f.done():
                            f.set_result(o)
                    self.stats["frames"] += len(items); self.stats["batches"] += 1
                    self.stats["batch_hist"][len(items)] += 1
                except Exception as e:                       # noqa: BLE001 - report to every waiter
                    self.stats["errors"] += 1
                    traceback.print_exc()
                    for f in futs:
                        if not f.done():
                            f.set_exception(e)
            except Exception:                                # noqa: BLE001 - never let the batcher die
                self.stats["errors"] += 1
                traceback.print_exc()
                await asyncio.sleep(0.01)

    @staticmethod
    def _key(r):
        return (float(r["cond_scale"]), bool(r.get("det", False)), tuple(int(x) for x in r["shape"]))

    def evict(self):
        cut = time.time() - self.state_ttl
        dead = [s for s, ent in list(self.states.items()) if ent["t"] < cut]
        for s in dead:
            self.states.pop(s, None)
        return len(dead)


def make_app(svc):
    async def act(req):
        try:
            r = await req.json()
            for k, typ in (("sid", str), ("text", str), ("nonce", str)):
                if not isinstance(r.get(k), typ):
                    raise ValueError(f"missing/invalid {k}")
            float(r["cond_scale"])
            h, w, c = (int(x) for x in r["shape"])
            img = np.frombuffer(base64.b64decode(r["image"]), dtype=np.uint8).reshape(h, w, c)
        except Exception as e:                           # noqa: BLE001
            return web.json_response({"error": f"bad request: {e}"}, status=400)
        r["img"] = img
        fut = asyncio.get_running_loop().create_future()
        svc.queue.append((r, fut)); svc.wake.set()
        try:
            out = await fut
        except Exception as e:                           # noqa: BLE001
            return web.json_response({"error": str(e)}, status=500)
        return web.json_response({"action": out["action"], "stale": out["stale"], "dup": out["dup"],
                                  "epoch": svc.epoch, "sid": r["sid"]})

    async def reset(req):
        r = await req.json()
        svc.states.pop(r.get("sid"), None)
        return web.json_response({"ok": True})

    async def health(req):
        h = svc.stats["batch_hist"]
        n = sum(h.values())
        alive = svc.batcher_task is not None and not svc.batcher_task.done()
        return web.json_response({"ok": alive, "epoch": svc.epoch, "device": svc.device, "states": len(svc.states),
                                  "frames": svc.stats["frames"], "batches": svc.stats["batches"],
                                  "mean_batch": (svc.stats["frames"] / n if n else 0.0),
                                  "errors": svc.stats["errors"], "state_miss": svc.stats["state_miss"],
                                  "dup_req": svc.stats["dup_req"], "t_gpu_s": round(svc.stats["t_gpu"], 1),
                                  "queue": len(svc.queue), "conds": len(svc.conds),
                                  "since_last_batch_s": round(time.time() - svc.stats["last_batch_t"], 1) if svc.stats["last_batch_t"] else None})

    def _died(task):
        if task.cancelled():
            return
        if task.exception() is not None:
            traceback.print_exception(task.exception())
            print("[steve] background task died -- exiting so the wrapper restarts the service", flush=True)
            os._exit(1)

    async def _start(app):
        svc.batcher_task = asyncio.create_task(svc.batcher())
        svc.batcher_task.add_done_callback(_died)

        async def _evict_loop():
            while True:
                await asyncio.sleep(120)
                try:
                    n = svc.evict()
                    if n:
                        print(f"[steve] evicted {n} idle chains ({len(svc.states)} live)", flush=True)
                except Exception:                        # noqa: BLE001
                    traceback.print_exc()
        app["evict"] = asyncio.create_task(_evict_loop())
        app["evict"].add_done_callback(_died)

    app = web.Application(client_max_size=8 * 1024 * 1024)
    app.on_startup.append(_start)
    app.add_routes([web.post("/act", act), web.post("/reset", reset), web.get("/health", health)])
    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=33500)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-wait-ms", type=float, default=4.0)
    ap.add_argument("--max-batch", type=int, default=64)
    ap.add_argument("--state-ttl-s", type=float, default=900.0)
    a = ap.parse_args()
    svc = SteveService(a.device, a.batch_wait_ms, a.max_batch, a.state_ttl_s)
    print(f"[steve] listening on :{a.port} batch_wait={a.batch_wait_ms}ms max_batch={a.max_batch} ttl={a.state_ttl_s}s", flush=True)
    web.run_app(make_app(svc), port=a.port, print=None)
