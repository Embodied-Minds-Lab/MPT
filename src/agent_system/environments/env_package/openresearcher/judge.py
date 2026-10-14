from __future__ import annotations

import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, Optional, Tuple

from .upstream import build_judge_prompt


def _first_match(patterns, text: str) -> Optional[str]:
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    return None


def parse_judge_response(judge_response: str) -> dict:
    """The judge's prose -> {"extracted_final_answer", "reasoning", "correct", "parse_error"}.
    `correct` is None (and parse_error True) when no verdict line was found: "we do not know" is
    NOT "wrong" -- callers count parse errors separately so a flaky judge cannot masquerade as a
    failing policy."""
    result = {"extracted_final_answer": None, "reasoning": None,
              "correct": None, "parse_error": False}
    if not judge_response or not isinstance(judge_response, str):
        result["parse_error"] = True
        return result

    result["extracted_final_answer"] = _first_match([
        r"\*\*extracted_final_answer:\*\*\s*(.*?)(?=\n|$)",
        r"\*\*extracted_final_answer\*\*:\s*(.*?)(?=\n|$)",
        r"extracted_final_answer:\s*(.*?)(?=\n|$)",
    ], judge_response)

    result["reasoning"] = _first_match([
        r"\*\*reasoning:\*\*\s*(.*?)(?=\n\*\*correct:\*\*|\n\*\*correct\*\*:|\ncorrect:|$)",
        r"\*\*reasoning\*\*:\s*(.*?)(?=\n\*\*correct:\*\*|\n\*\*correct\*\*:|\ncorrect:|$)",
        r"reasoning:\s*(.*?)(?=\ncorrect:|$)",
    ], judge_response)

    verdict = _first_match([
        r"(?m)^\s*\*\*correct:\*\*\s*(yes|no)\b",
        r"(?m)^\s*\*\*correct\*\*:\s*(yes|no)\b",
        r"(?m)^\s*correct:\s*(yes|no)\b",
        # qwen3.5-flash decorates the verdict a few % of the time: `correct: \boxed{no}`, `correct: **yes**`,
        # `correct: "no"` (09-23: 3-7 "malformed judge reply" per 64 episodes, each scored as judge_error = reward 0)
        r"(?m)^\s*\**correct:?\**:?\s*(?:\\boxed\{|\*\*|[\"'`]|\$)*\s*(yes|no)\b",
    ], judge_response)
    if verdict is not None:
        result["correct"] = verdict.lower() == "yes"
    else:
        result["parse_error"] = True
    return result


def failed_verdict(reason: str) -> Dict[str, Any]:
    return {"extracted_final_answer": None, "reasoning": None, "correct": None,
            "parse_error": True, "call_error": True, "error": reason}


class JudgeDownError(BaseException):
    """The judge endpoint failed `max_consecutive_failures` grades in a row.

    BaseException for the same reason as service.ServiceDownError: a failed verdict scores the
    episode 0 (envs.py), so a dead judge silently turns a whole batch into "every answer was
    wrong" and REINFORCE trains on it. Better to abort; resume_mode=auto continues from the last
    checkpoint once the endpoint is back."""



def _not_retryable(exc: BaseException) -> bool:
    """4xx replies the API returns deterministically (content filter, bad request): no point retrying."""
    try:
        from agent_system.environments.env_package.openresearcher.api_client import not_retryable
        return not_retryable(exc)
    except Exception:  # noqa: BLE001 -- offline stubs without api_client
        return "DataInspectionFailed" in str(exc) or "BadRequestError" in str(exc) or "NonRetryable" in type(exc).__name__

class Judge:
    """One judge per env pool. Thread-safe: env.step() runs in a thread per env.

    `client` must expose `generate(prompt: str) -> str`. Built from `env.openresearcher.judge`
    via `build_judge(cfg)` in production; tests pass a stub.

    Cache: keyed by (qid, whitespace/case-normalised answer). Rollouts in a GRPO group converge on
    the same answer string, and the same question recurs across epochs, so this saves real judge
    calls. Failures are never cached (a transient outage must not become a permanent 0).

    `max_consecutive_failures` (0 = off): after that many grades in a row ended in
    failed_verdict, raise JudgeDownError instead of handing back another reward-0 verdict.
    """

    def __init__(self, client, num_retries: int = 3, max_concurrency: int = 16,
                 cache_max: int = 50000, name: str = "judge", max_consecutive_failures: int = 0):
        self.client = client
        self.num_retries = max(int(num_retries), 1)
        self._sem = threading.BoundedSemaphore(max(int(max_concurrency), 1))
        self._max_concurrency = max(int(max_concurrency), 1)
        self._cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._cache_max = int(cache_max)
        self._lock = threading.Lock()
        self.name = name
        self.n_calls = 0
        self.n_failures = 0
        self.max_consecutive_failures = max(int(max_consecutive_failures), 0)
        self._consecutive_failures = 0

    @staticmethod
    def _key(qid: str, answer: str) -> Tuple[str, str]:
        return (str(qid), " ".join(str(answer).lower().split()))

    def grade(self, qid: str, question: str, answer: str, correct_answer: str) -> Dict[str, Any]:
        key = self._key(qid, answer)
        with self._lock:
            hit = self._cache.get(key)
        if hit is not None:
            return dict(hit)
        prompt = build_judge_prompt(question, str(answer), str(correct_answer))
        last = "malformed judge reply"
        last_reply = ""           # head of the last unparseable reply, printed with the failure (a malformed
                                  # judge reply is otherwise invisible: up to ~10 % of grades were seen to fail so)
        verdict: Optional[Dict[str, Any]] = None
        with self._sem:
            for _attempt in range(self.num_retries):
                try:
                    with self._lock:
                        self.n_calls += 1
                    reply = self.client.generate(prompt)
                    v = parse_judge_response(reply)
                    if v.get("correct") is not None and not v.get("parse_error"):
                        v["call_error"] = False
                        verdict = v
                        break
                    last = "malformed judge reply"
                    last_reply = str(reply)
                except Exception as exc:        # noqa: BLE001 -- retried, then reported
                    last = f"{type(exc).__name__}: {exc}"
                    if _not_retryable(exc):
                        break                   # the API answers this question the same way every time
        if verdict is None:
            with self._lock:
                self.n_failures += 1
                self._consecutive_failures += 1
                streak = self._consecutive_failures
            print(f"[openresearcher-{self.name}] qid={qid} failed after {self.num_retries} "
                  f"attempts: {last} (consecutive failures: {streak})"
                  + (f" reply_len={len(last_reply)} reply_head={last_reply[:240]!r} reply_tail={last_reply[-160:]!r}"
                     if last_reply else ""), flush=True)
            if self.max_consecutive_failures and streak >= self.max_consecutive_failures:
                msg = (f"[openresearcher-{self.name}] {streak} consecutive judge failures (last: {last}) -- "
                       f"run ABORTED so no episode is trained on a reward the judge never produced. Check "
                       f"JUDGE_API_BASE / JUDGE_API_KEY / the model name, then resubmit (resume_mode=auto).")
                print(msg, flush=True)
                raise JudgeDownError(msg)
            return failed_verdict(last)
        with self._lock:
            self._consecutive_failures = 0
            if len(self._cache) >= self._cache_max:
                self._cache.clear()
            self._cache[key] = dict(verdict)
        return dict(verdict)

    def grade_many(self, items: Iterable[Tuple[str, str, str, str]], max_workers: Optional[int] = None):
        items = list(items)
        with ThreadPoolExecutor(max_workers=max_workers or self._max_concurrency) as ex:
            return list(ex.map(lambda t: self.grade(*t), items))


class ExactMatchClient:
    """Offline stand-in for the LLM judge (smoke runs, CPU tests): answers with a verdict block
    the parser accepts, 'yes' iff the response equals the correct answer after normalisation.
    Requires the prompt built by build_judge_prompt (it reads the [response]/[correct_answer]
    sections back out)."""
    _RESP = re.compile(r"\[response\]: (.*?)\n\nYour judgement", re.DOTALL)
    _GOLD = re.compile(r"\[correct_answer\]: (.*?)\n\nreasoning:", re.DOTALL)

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        r = self._RESP.search(prompt)
        g = self._GOLD.search(prompt)
        resp = " ".join((r.group(1) if r else "").lower().split())
        gold = " ".join((g.group(1) if g else "").lower().split())
        ok = bool(resp) and resp == gold
        return (f"extracted_final_answer: {r.group(1).strip() if r else 'None'}\n"
                f"reasoning: offline exact-match evaluator\n"
                f"correct: {'yes' if ok else 'no'}\n"
                f"confidence: 100")


def build_judge(jcfg, name: str = "judge") -> Judge:
    """From the `env.openresearcher.judge` config block (any mapping with .get)."""
    def _g(k, d=None):
        try:
            v = jcfg.get(k, d)
        except AttributeError:
            v = getattr(jcfg, k, d)
        return d if v is None else v

    api = str(_g("api", "litellm")).lower()
    if api in ("offline", "exact_match"):
        client = ExactMatchClient()
    else:
        from agent_system.environments.env_package.openresearcher.api_client import OpenAICompatClient
        client = OpenAICompatClient(
            model=str(_g("model", "claude-sonnet-5")),
            api_base=str(_g("api_base", "")) or None,
            api_key=str(_g("api_key", "")) or None,
            api_base_env=str(_g("api_base_env", "JUDGE_API_BASE")),
            api_key_env=str(_g("api_key_env", "JUDGE_API_KEY")),
            max_tokens=int(_g("max_tokens", 4096)),
            temperature=float(_g("temperature", 0.0)),
            top_p=_g("top_p", None),
            timeout=float(_g("timeout", 180.0)),
            max_retries=int(_g("max_retries", 4)),
            base_delay=float(_g("base_delay", 1.0)),
            bypass_proxy=bool(_g("bypass_proxy", True)),
            drop_params=bool(_g("drop_params", True)),
        )
    return Judge(client, num_retries=int(_g("num_retries", 3)),
                 max_concurrency=int(_g("max_concurrency", 16)), name=name,
                 max_consecutive_failures=int(_g("max_consecutive_failures", 0)))
