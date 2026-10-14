from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import dataclasses
import datetime as _dt
import json
import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .action import decode_action, decode_hit_cap
from .browser import BrowserState
from .judge import Judge, build_judge
from .parser import STOP_STRINGS, cut_at_stop, has_answer_marker, parse_assistant_turn
from .service import DEFAULT_MAX_RETRIES, ServiceClient
from .tools import execute_tool
from .upstream import DEVELOPER_CONTENT, TOOL_CONTENT

TOOLS: List[Dict[str, Any]] = json.loads(TOOL_CONTENT)

# OpenResearcher has no text for an unparseable <tool_call> (their loop appends an assistant turn
# with no tool call and generates again). We must hand the model SOMETHING as the tool result so
# the conversation shape stays assistant/tool alternating; this is the only wording that is ours.
PARSE_ERROR_TEXT = ("Error: could not parse the tool call. Emit exactly one "
                    "<tool_call>{\"name\": ..., \"arguments\": {...}}</tool_call> block.")


def system_prompt(date: str) -> str:
    """deploy_agent.py:193: DEVELOPER_CONTENT + the date line, no other bytes."""
    return f"{DEVELOPER_CONTENT}\n\nToday's date: {date}"


@dataclass
class EnvSettings:
    date: str
    require_answer_marker: bool = False
    trace_path: str = ""
    is_train: bool = True
    # A turn that opened <think> or <tool_call> and never closed it is a generation cut off at
    # data.max_response_length (typically a repetition loop). False (default): such a turn ends the
    # episode with reward 0, is NOT sent to the judge (it would otherwise waste judge calls on
    # garbage) and is reported as `truncated`, not `answered`. True: the earlier rule -- the whole
    # text is graded as the answer.
    truncated_turn_is_answer: bool = False
    # A think-only turn (no tool call, nothing after </think>) is a burned turn, not an answer.
    # True restores the earlier rule (graded as an empty answer, episode over) -- used to evaluate a
    # checkpoint under the exact rules it was trained with.
    think_only_is_answer: bool = False


def is_truncated_turn(text: str) -> bool:
    """<think> / <tool_call> opened but never closed = the response hit the length cap."""
    t = text or ""
    return ("<think>" in t and "</think>" not in t) or ("<tool_call>" in t and "</tool_call>" not in t)


class OpenResearcherEnv:
    def __init__(self, client: ServiceClient, judge: Judge, settings: EnvSettings):
        self.client = client
        self.judge = judge
        self.settings = settings
        self.qid = ""
        self.question = ""
        self.answer = ""
        self.data_source = "openresearcher"
        self.dummy = True
        self.done = True
        self.round = 0
        self.browser = BrowserState()
        self.messages: List[Dict[str, Any]] = []
        self.tool_counts: Dict[str, int] = {}
        self._graded: Optional[Dict[str, Any]] = None

    # ---- episode lifecycle -------------------------------------------------------------------
    def reset(self, extras: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if self.messages and not self.done and not self.dummy:
            self._trace(finished=False)          # the previous episode hit the horizon unanswered
        self.qid = str(extras.get("qid", ""))
        self.question = str(extras.get("question", ""))
        self.answer = str(extras.get("answer", ""))
        self.data_source = str(extras.get("data_source", "openresearcher"))
        self.dummy = (self.question == "")
        self.done = self.dummy
        self.round = 0
        self.browser = BrowserState()
        self.tool_counts = {}
        self._graded = None
        self.messages = [{"role": "system", "content": system_prompt(self.settings.date)},
                         {"role": "user", "content": self.question}]
        return copy.deepcopy(self.messages), self._info()

    def _info(self, **kw) -> Dict[str, Any]:
        # A finished episode keeps reporting its terminal verdict: the chain's finalize rows and any
        # step() on a done env take this default path, and success_evaluator reads `won` /
        # `truncated` / `judge_error` off the LAST row -- with plain False defaults the chain batch line
        # printed correct=0 (and pass@k under-counted) while the env trace held 3-7 correct per
        # iteration (the reward stream was never affected).
        g = self._graded if self.done else None
        v = (g or {}).get("verdict") or {}
        info = {"data_source": self.data_source, "qid": self.qid, "won": bool(v.get("correct")),
                "tool_calling": False, "is_action_valid": True, "tool_name": "",
                "whitelist_ok": True, "n_rounds": self.round, "done": self.done,
                "truncated": bool((g or {}).get("truncated", False)),
                "judge_error": bool(v.get("call_error") or v.get("parse_error"))}
        info.update(kw)
        return info

    def step(self, action: str):
        if self.dummy or self.done:
            return copy.deepcopy(self.messages), 0.0, True, self._info()
        text, allowed = decode_action(action)
        hit_cap = decode_hit_cap(action)
        text = cut_at_stop(text)
        self.round += 1
        parsed = parse_assistant_turn(text)
        # the RAW turn goes into history: the chat template re-renders exactly what the model wrote
        self.messages.append({"role": "assistant", "content": text})

        # "truncated" = the generation really filled the response budget (the agent's hit_cap flag). The
        # textual heuristic (unclosed <think>/<tool_call>) stays only for actions that carry no flag;
        # measured: 80 % of the turns it flagged had ended normally -- a JSON-complete tool call
        # missing </tool_call> (now parsed) or a short open <think> (now graded as the answer it is).
        truncated = hit_cap if hit_cap is not None else is_truncated_turn(text)
        if not self.settings.truncated_turn_is_answer and not parsed.has_tool_call and truncated:
            # cut off at the response cap: not an answer, not a tool call. Episode over, reward 0,
            # no judge call; the trace and the batch line count it separately from `answered`.
            self.done = True
            self._graded = {"answer": parsed.answer_text, "verdict": None, "truncated": True}
            info = self._info(tool_name="", truncated=True, is_action_valid=False, answer=parsed.answer_text)
            self._trace(finished=False, truncated=True)
            return copy.deepcopy(self.messages), 0.0, True, info

        if parsed.has_tool_call:
            if parsed.parse_error:
                result = PARSE_ERROR_TEXT
                info = self._info(is_action_valid=False, tool_name=str(parsed.name or ""))
            else:
                name = str(parsed.name or "")
                result = execute_tool(name, parsed.args, self.browser, self.client, allowed=allowed)
                unavailable = result == f"Tool {name} not available"
                self.tool_counts[name] = self.tool_counts.get(name, 0) + (0 if unavailable else 1)
                info = self._info(tool_calling=not unavailable, tool_name=name,
                                  whitelist_ok=not unavailable, is_action_valid=not unavailable)
            # tool turns carry no reward; the answer turn is graded by the judge
            self.messages.append({"role": "tool", "tool_call_id": str(self.round), "content": result})
            return copy.deepcopy(self.messages), 0.0, False, info

        # no tool call: the turn is the final answer (OR's own marker rule available behind a flag)
        if self.settings.require_answer_marker and not has_answer_marker(parsed.content):
            return copy.deepcopy(self.messages), 0.0, False, self._info(tool_name="")
        answer_text = parsed.answer_text
        if not answer_text.strip() and not self.settings.think_only_is_answer:
            # A turn with no tool call AND nothing after </think> (a think-only turn -- common on
            # first turns, often with the query written inside the think) is NOT a final answer. Treat it like an unparseable tool call: the turn is burned, the episode
            # continues, no judge call. Nothing inside the think is executed.
            self.messages.append({"role": "tool", "tool_call_id": str(self.round), "content": PARSE_ERROR_TEXT})
            return copy.deepcopy(self.messages), 0.0, False, self._info(is_action_valid=False, tool_name="", think_only=True)
        verdict = self.judge.grade(self.qid, self.question, answer_text, self.answer)
        self.done = True
        won = bool(verdict.get("correct"))
        judge_error = bool(verdict.get("call_error")) or bool(verdict.get("parse_error"))
        self._graded = {"answer": answer_text, "verdict": verdict}
        info = self._info(won=won, tool_name="", answer=answer_text, judge_error=judge_error,
                          judge_parse_error=bool(verdict.get("parse_error")),
                          judge_extracted=verdict.get("extracted_final_answer"))
        self._trace(finished=True)
        return copy.deepcopy(self.messages), float(won), True, info

    def close(self) -> None:
        pass

    # ---- counterfactual snapshot (episode state only) ------------------------------------------
    def export_state(self) -> Dict[str, Any]:
        """Everything episode-local, deep-copied: the message list, the browser (page cache +
        page stack), the round and tool counters, done and the grading. The search client, the
        judge and the settings are NOT part of it: the search service is stateless, and the
        judge cache (keyed by qid and answer) is safe to share."""
        return {"qid": self.qid, "question": self.question, "answer": self.answer,
                "data_source": self.data_source, "dummy": self.dummy, "done": self.done,
                "round": self.round, "messages": copy.deepcopy(self.messages),
                "browser": copy.deepcopy(self.browser.export_state()),
                "tool_counts": dict(self.tool_counts), "graded": copy.deepcopy(self._graded)}

    def load_state(self, st: Dict[str, Any]) -> None:
        """Restore an export_state() snapshot onto THIS env (the browser is restored in place)."""
        self.qid = str(st.get("qid", ""))
        self.question = str(st.get("question", ""))
        self.answer = str(st.get("answer", ""))
        self.data_source = str(st.get("data_source", "openresearcher"))
        self.dummy = bool(st.get("dummy", False))
        self.done = bool(st.get("done", False))
        self.round = int(st.get("round", 0))
        self.messages = copy.deepcopy(list(st.get("messages") or []))
        self.browser.restore_state(st.get("browser") or {})
        self.tool_counts = dict(st.get("tool_counts") or {})
        self._graded = copy.deepcopy(st.get("graded"))

    # ---- trace -------------------------------------------------------------------------------
    def _trace(self, finished: bool, truncated: bool = False) -> None:
        path = self.settings.trace_path
        if not path:
            return
        rec = {"qid": self.qid, "data_source": self.data_source, "train": self.settings.is_train,
               "finished": finished, "truncated": truncated, "rounds": self.round, "tool_counts": dict(self.tool_counts),
               "answer": (self._graded or {}).get("answer"), "gold": self.answer,
               "verdict": (self._graded or {}).get("verdict"), "messages": self.messages,
               "ts": _dt.datetime.now().isoformat(timespec="seconds"), "pid": os.getpid()}
        try:
            with open(path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        except Exception as exc:                # noqa: BLE001 -- tracing must never kill a rollout
            print(f"[openresearcher] trace write failed (non-fatal): {exc}", flush=True)


class OpenResearcherMultiProcessEnv:
    """Thread-parallel wrapper (same contract as MathMultiProcessEnv). One ServiceClient and one
    Judge shared by every env of the pool; the client probes the service at construction."""

    def __init__(self, seed: int = 0, env_num: int = 1, group_n: int = 1, is_train: bool = True,
                 service_url: str = "", settings: EnvSettings = None, judge: Judge = None,
                 service_timeout: float = 60.0, service_max_retries: int = DEFAULT_MAX_RETRIES,
                 service_url_file: str | None = None, n_spare: int = 0):
        self.env_num = env_num
        self.group_n = group_n
        self.batch_size = env_num * group_n
        self.is_train = is_train
        self.seed = seed
        self.service_url = service_url
        self.client = ServiceClient(service_url, timeout=service_timeout, max_retries=service_max_retries,
                                    url_file=service_url_file)
        print(f"[openresearcher] {'train' if is_train else 'val'} pool: service={service_url} "
              f"(active_searcher={self.client.active_searcher}) batch={self.batch_size}", flush=True)
        self.judge = judge
        self.settings = settings or EnvSettings(date=_dt.date.today().strftime("%Y-%m-%d"), is_train=is_train)
        self.envs = [OpenResearcherEnv(self.client, self.judge, self.settings) for _ in range(self.batch_size)]
        # Spare envs for the counterfactual branches (n_spare=2 when training with
        # cf_preference). They live OUTSIDE self.envs, so the mainline reset/step never touches
        # them, and they write no OR_TRACE records (a branch is not an episode).
        _spare_settings = dataclasses.replace(self.settings, trace_path="")
        self.spare_envs = [OpenResearcherEnv(self.client, self.judge, _spare_settings)
                           for _ in range(int(n_spare))]
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=min(max(self.batch_size, 1), 256))
        self._loop = asyncio.new_event_loop()
        self._closed = False

    # ---- helpers -----------------------------------------------------------------------------
    @staticmethod
    def _sync_reset(env: OpenResearcherEnv, kwargs: Dict[str, Any]):
        extras = {"qid": kwargs.get("qid", ""), "question": kwargs.get("question", ""),
                  "answer": kwargs.get("answer", kwargs.get("ground_truth", "")),
                  "data_source": kwargs.get("data_source", "openresearcher")}
        return env.reset(extras)

    @staticmethod
    def _sync_step(env: OpenResearcherEnv, action: str):
        return env.step(action)

    def _run(self, fn, envs, args):
        tasks = [self._loop.run_in_executor(self._executor, fn, env, a) for env, a in zip(envs, args)]
        return self._loop.run_until_complete(asyncio.gather(*tasks))

    # ---- contract ----------------------------------------------------------------------------
    def reset(self, kwargs: List[Dict[str, Any]]):
        if len(kwargs) > self.batch_size:
            raise ValueError(f"Got {len(kwargs)} kwarg dicts, but the env was initialised with "
                             f"total_envs={self.batch_size}")
        pad_n = self.batch_size - len(kwargs)
        dummy = {"qid": "", "question": "", "answer": "", "data_source": "openresearcher"}
        padded = list(kwargs) + [dummy] * pad_n
        results = self._run(self._sync_reset, self.envs, padded)
        obs_list = [o for o, _ in results][:len(kwargs)]
        info_list = [i for _, i in results][:len(kwargs)]
        return obs_list, info_list

    def step(self, actions: List[str]):
        if len(actions) > self.batch_size:
            raise ValueError(f"Got {len(actions)} actions, but the env was initialized with "
                             f"total_envs={self.batch_size}")
        pad_n = self.batch_size - len(actions)
        padded = list(actions) + [""] * pad_n
        results = self._run(self._sync_step, self.envs, padded)
        n = len(actions)
        obs_list = [r[0] for r in results][:n]
        reward_list = [r[1] for r in results][:n]
        done_list = [r[2] for r in results][:n]
        info_list = [r[3] for r in results][:n]
        return obs_list, reward_list, done_list, info_list

    # ---- counterfactual env adapter (the collector's cf_* contract; see
    # ---- multi_turn_rollout/counterfactual_collector.py) --------------------------------------
    # In process: a snapshot is the mainline env's exported episode state; a branch slot is a
    # spare env (slot batch_size + j -> self.spare_envs[j]) restored from it.
    def cf_bind(self, timeout: float = None) -> List[int]:
        if len(self.spare_envs) < 2:
            raise RuntimeError(
                f"cf collector: the deep-research env pool has {len(self.spare_envs)} spare envs; "
                "the counterfactual bid phase needs 2 (built by make_envs when training with "
                "agent.orchestra.openresearcher_chain.cf_preference=True)")
        return [self.batch_size, self.batch_size + 1]

    def _spare(self, slot: int) -> OpenResearcherEnv:
        j = int(slot) - self.batch_size
        if not 0 <= j < len(self.spare_envs):
            raise RuntimeError(f"slot {slot} is not a spare env slot "
                               f"([{self.batch_size}, {self.batch_size + len(self.spare_envs)}))")
        return self.spare_envs[j]

    def cf_snapshot(self, slot: int) -> Dict[str, Any]:
        if not 0 <= int(slot) < self.batch_size:
            return {"error": f"slot {slot} is not a mainline slot"}
        return self.envs[int(slot)].export_state()

    def cf_restore(self, slot: int, snap: Dict[str, Any]) -> Dict[str, Any]:
        self._spare(slot).load_state(copy.deepcopy(snap))
        return {"ok": True}

    def cf_step_slot(self, slot: int, action: str) -> Dict[str, Any]:
        obs, reward, done, info = self._spare(slot).step(action)
        # a branch scored on a judge failure would compare a real verdict with a missing one
        if info.get("judge_error"):
            raise RuntimeError(f"branch judge call failed on slot {slot} (qid={info.get('qid')})")
        return {"messages": obs, "reward": float(reward), "done": bool(done), "info": info}

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for env in self.envs + list(getattr(self, "spare_envs", [])):
            env.close()
        self._executor.shutdown(wait=False)
        try:
            self._loop.close()
        except Exception:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ---- construction from config -----------------------------------------------------------------
def _read_url_file(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    try:
        with open(path) as f:
            return f.read().strip() or None
    except OSError:
        return None


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    try:
        v = cfg.get(key, default)
    except AttributeError:
        v = getattr(cfg, key, default)
    return default if v is None else v


def resolve_service_url_file(ocfg, is_train: bool) -> str | None:
    """The address file the matching service launcher maintains (failover target), same val_corpus rule."""
    if not is_train and str(_cfg_get(ocfg, "val_corpus", "train") or "train").strip().lower() == "train":
        is_train = True
    return os.environ.get("OR_SERVICE_URL_FILE" if is_train else "OR_SERVICE_URL_VAL_FILE") or None


def resolve_service_url(ocfg, is_train: bool) -> str:
    """Config value -> env var -> address file written by the service's launcher. With
    val_corpus=val, validation falls back to the TRAIN service with a loud warning (then it browses
    the wrong corpus)."""
    if not is_train and str(_cfg_get(ocfg, "val_corpus", "train") or "train").strip().lower() == "train":
        # val_or (held-out OpenResearcher questions) must browse the OpenResearcher corpus, not BCP
        print("[openresearcher] val pool: env.openresearcher.val_corpus=train -> the validation pool browses "
              "the OpenResearcher-corpus service (held-out OpenResearcher questions); a BrowseComp-Plus "
              "evaluation needs val_corpus=val", flush=True)
        return resolve_service_url(ocfg, True)
    if is_train:
        cands = [_cfg_get(ocfg, "service_url", ""), os.environ.get("OR_SERVICE_URL", ""),
                 _read_url_file(os.environ.get("OR_SERVICE_URL_FILE"))]
    else:
        cands = [_cfg_get(ocfg, "service_url_val", ""), os.environ.get("OR_SERVICE_URL_VAL", ""),
                 _read_url_file(os.environ.get("OR_SERVICE_URL_VAL_FILE"))]
    url = next((str(c).strip() for c in cands if c and str(c).strip()), None)
    if url is None and not is_train:
        print("[openresearcher] WARNING: no validation service URL (env.openresearcher.service_url_val / "
              "OR_SERVICE_URL_VAL / OR_SERVICE_URL_VAL_FILE); validation will use the TRAIN corpus "
              "service. BrowseComp-Plus questions need the BCP corpus -- fix this before trusting "
              "val numbers.", flush=True)
        return resolve_service_url(ocfg, True)
    if url is None:
        raise RuntimeError("OpenResearcher search service URL not set: pass env.openresearcher.service_url, "
                           "or export OR_SERVICE_URL, or OR_SERVICE_URL_FILE (the address file the "
                           "service job writes). See scripts/start_or_search_service.sh.")
    return url.rstrip("/")


_JUDGES: Dict[str, Judge] = {}
_JUDGE_LOCK = threading.Lock()


def build_openresearcher_envs(seed: int = 0, env_num: int = 1, group_n: int = 1, is_train: bool = True,
                              env_config=None, n_spare: int = 0):
    ocfg = _cfg_get(env_config, "openresearcher", None)
    date = str(_cfg_get(ocfg, "date", "") or "").strip() or _dt.date.today().strftime("%Y-%m-%d")
    settings = EnvSettings(date=date,
                           require_answer_marker=bool(_cfg_get(ocfg, "require_answer_marker", False)),
                           trace_path=os.environ.get("OR_TRACE", ""),
                           is_train=is_train,
                           truncated_turn_is_answer=bool(_cfg_get(ocfg, "truncated_turn_is_answer", False)),
                           think_only_is_answer=bool(_cfg_get(ocfg, "think_only_is_answer", False)))
    url = resolve_service_url(ocfg, is_train)
    # one judge per split (separate caches; same config), built once per process
    tag = "judge-train" if is_train else "judge-val"
    with _JUDGE_LOCK:
        if tag not in _JUDGES:
            _JUDGES[tag] = build_judge(_cfg_get(ocfg, "judge", {}) or {}, name=tag)
        judge = _JUDGES[tag]
    return OpenResearcherMultiProcessEnv(
        seed=seed, env_num=env_num, group_n=group_n, is_train=is_train, service_url=url,
        settings=settings, judge=judge,
        service_timeout=float(_cfg_get(ocfg, "service_timeout", 60.0)),
        service_max_retries=int(_cfg_get(ocfg, "service_max_retries", DEFAULT_MAX_RETRIES)),
        service_url_file=resolve_service_url_file(ocfg, is_train), n_spare=n_spare)


__all__ = ["OpenResearcherEnv", "OpenResearcherMultiProcessEnv", "build_openresearcher_envs",
           "resolve_service_url", "EnvSettings", "TOOLS", "STOP_STRINGS", "system_prompt", "is_truncated_turn"]
