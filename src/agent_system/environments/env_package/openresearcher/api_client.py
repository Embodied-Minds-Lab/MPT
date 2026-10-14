from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional


def bypass_proxy() -> None:
    os.environ["no_proxy"] = "*"
    os.environ["NO_PROXY"] = "*"



class NonRetryableError(RuntimeError):
    """A request the API rejected deterministically (4xx content filter / bad request); callers should not retry."""


_NON_RETRYABLE = ("DataInspectionFailed", "BadRequestError", "inappropriate", "content_filter", "ContentPolicy",
                  "invalid_request_error", "context_length_exceeded")


def not_retryable(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}"
    return any(k in text for k in _NON_RETRYABLE)

class OpenAICompatClient:
    def __init__(self, model: str, api_base: Optional[str] = None, api_key: Optional[str] = None,
                 api_base_env: str = "JUDGE_API_BASE", api_key_env: str = "JUDGE_API_KEY",
                 max_tokens: int = 4096, temperature: float = 0.0, top_p: Optional[float] = None,
                 timeout: float = 180.0, max_retries: int = 4, base_delay: float = 1.0,
                 empty_retries: int = 3, bypass_proxy: bool = True, drop_params: bool = True,
                 send_sampling: bool = True, extra: Optional[Dict[str, Any]] = None):
        if not model:
            raise ValueError("model name is required")
        self.model = model if "/" in model else f"openai/{model}"
        self.api_base = (api_base or os.environ.get(api_base_env) or os.environ.get("OPENAI_BASE_URL") or "").strip() or None
        self.api_key = (api_key or os.environ.get(api_key_env) or os.environ.get("OPENAI_API_KEY") or "").strip() or None
        if self.api_base is None:
            raise RuntimeError(f"no API base URL: pass api_base or export {api_base_env} (or OPENAI_BASE_URL)")
        if self.api_key is None:
            raise RuntimeError(f"no API key: export {api_key_env} (or OPENAI_API_KEY); never put keys in config files")
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.top_p = None if top_p is None else float(top_p)
        self.timeout = float(timeout)
        self.max_retries = max(int(max_retries), 1)
        self.base_delay = float(base_delay)
        self.empty_retries = max(int(empty_retries), 1)
        self.drop_params = bool(drop_params)
        self.send_sampling = bool(send_sampling)
        self.extra = dict(extra or {})
        if bypass_proxy:
            globals()["bypass_proxy"]()
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)
        self._lock = threading.Lock()
        self.n_calls = 0
        self.n_errors = 0

    # ---- core --------------------------------------------------------------------------------
    def _request(self, messages: List[Dict[str, str]], **overrides) -> Dict[str, Any]:
        req: Dict[str, Any] = {"model": self.model, "messages": messages, "max_tokens": self.max_tokens,
                               "timeout": self.timeout, "drop_params": self.drop_params,
                               "api_base": self.api_base, "api_key": self.api_key}
        if self.send_sampling:
            req["temperature"] = self.temperature
            if self.top_p is not None:
                req["top_p"] = self.top_p
        req.update(self.extra)
        req.update(overrides)
        return req

    def chat(self, messages: List[Dict[str, str]], **overrides) -> str:
        """One logical call -> the reply text ('' only after every retry produced nothing)."""
        from litellm import completion   # lazy: litellm is slow to import and not needed offline
        req = self._request(messages, **overrides)
        for e_try in range(self.empty_retries):
            last: Optional[BaseException] = None
            resp = None
            for attempt in range(self.max_retries):
                try:
                    with self._lock:
                        self.n_calls += 1
                    resp = completion(**req)
                    break
                except Exception as exc:        # noqa: BLE001 -- retried, then raised
                    last = exc
                    with self._lock:
                        self.n_errors += 1
                    if not_retryable(exc):
                        # a 4xx the server will answer the same way every time (content filter
                        # "DataInspectionFailed", bad request): fail now. Retrying with backoff
                        # (x max_retries here x num_retries in the judge) stalled a whole env step
                        # for 200-640 s per such question.
                        raise NonRetryableError(str(exc)) from exc
                    if attempt < self.max_retries - 1:
                        time.sleep(self.base_delay * (2 ** attempt))
            if resp is None:
                raise RuntimeError(f"litellm completion failed after {self.max_retries} attempts: {last}")
            content = self._content_of(resp)
            if content.strip():
                return content
            if e_try < self.empty_retries - 1:
                time.sleep(2.0 * (e_try + 1))
        return ""

    @staticmethod
    def _content_of(resp) -> str:
        try:
            choice = (resp.choices or [None])[0]
            if choice is None:
                return ""
            msg = getattr(choice, "message", None)
            content = getattr(msg, "content", None) if msg is not None else None
            if content is None and isinstance(choice, dict):
                content = (choice.get("message") or {}).get("content")
            return content or ""
        except Exception:                        # noqa: BLE001 -- treat as empty, retried above
            return ""

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **overrides) -> str:
        msgs: List[Dict[str, str]] = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": prompt})
        return self.chat(msgs, **overrides)

    def __repr__(self) -> str:
        return (f"OpenAICompatClient(model={self.model!r}, api_base={self.api_base!r}, "
                f"max_tokens={self.max_tokens}, temperature={self.temperature}, top_p={self.top_p})")


__all__ = ["OpenAICompatClient", "bypass_proxy"]
