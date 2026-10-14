from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests


DEFAULT_MAX_RETRIES = 3
RETRY_SLEEP_S = 1.0


class BackendError(Exception):
    """The search service ANSWERED with a failure (4xx, empty content, no
    hits) — reported to the model as tool text, same as upstream's
    BackendError. Deterministic answers are never retried."""


class ServiceDownError(BaseException):
    """The search service stayed unreachable through the whole retry budget.

    BaseException ON PURPOSE (the KeyboardInterrupt trick): every layer
    between a tool and main() — the tool error-wrapper, smolagents' agent
    loop, the framework's per-agent guards — catches Exception to keep an
    episode alive, and this must keep NOTHING alive. A dead service means
    every further LLM turn is spent browsing a void, so the run aborts
    immediately instead of finishing with silently degraded accuracy.
    Restart the service (README.md, 4. Evaluation, Deep Research) and rerun the same
    config: resume skips finished examples and rebuilds the interrupted one
    at its last checkpoint."""


class ServiceClient:
    """OpenResearcher's search service, spoken to synchronously.

    The service (their scripts/deploy_search_service.py, run unmodified) is
    started BY THE USER, once, and shared by every concurrent run (README.md, 4. Evaluation, Deep Research);
    this client only talks to it: ``POST /search {query, topn}`` ->
    {"results": [{url, title, summary}]} and ``POST /get_content {url}`` ->
    {title, content}. Which retriever answers (bm25/dense) was decided at
    service launch; ``active_searcher`` records what this run actually got.

    ``search`` accepts a string or a LIST of queries — OpenResearcher's
    extension to the gpt-oss browser (their backend fans a list out in
    parallel and concatenates the hits in query order); the fan-out here is
    threads instead of their asyncio, results ordered identically.
    """

    def __init__(self, service_url: str, timeout: float = 60.0,
                 max_retries: int = DEFAULT_MAX_RETRIES, url_file: str | None = None):
        self._base = service_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        # The service launcher's address file (OR_SERVICE_URL[_VAL]_FILE). A long-running service may
        # be restarted elsewhere; a trainer that cached the old URL at start would abort with
        # ServiceDownError the moment the old service went away. A transport failure therefore
        # re-reads the file and fails over to the new address before the retry budget is spent.
        self._url_file = url_file or None
        # The service is an INTERNAL endpoint (started next to the trainer, see
        # scripts/start_or_search_service.sh) and an ambient HTTP(S)_PROXY cannot
        # reach it, so this client never consults proxy env vars: the same default-bypass
        # convention as llm.bypass_proxy, hardcoded because unlike a model
        # API there is no proxied-service deployment to support.
        self._http = requests.Session()
        self._http.trust_env = False
        # one pooled connection per concurrent episode: the env drives up to 256 episodes at once
        # (envs.py executor) and urllib3's default pool of 10 logs "Connection pool is full,
        # discarding connection" for every extra one -- thousands of lines per hour of a
        # run. Harmless, but it buries the lines that matter.
        _adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=256)
        self._http.mount("http://", _adapter)
        self._http.mount("https://", _adapter)
        # probe NOW, not on the first mid-run search: an unreachable service
        # should fail the run before any agent has spent a single LLM call
        self.active_searcher = self._probe()

    def _request(self, method: str, path: str, **kwargs):
        """One service HTTP call under the retry budget.

        Retried (up to ``max_retries`` attempts, ``RETRY_SLEEP_S`` apart):
        transport failures and 5xx answers — the shapes a dying/restarting
        service produces. NOT retried: any other answer (200, 404, ...) —
        the service is SPEAKING, and its answers are deterministic. A budget
        that runs dry raises ServiceDownError, which aborts the whole run.
        """
        last = "no attempt made"
        attempt, budget, switched = 0, self._max_retries, False
        while attempt < budget:
            attempt += 1
            try:
                resp = self._http.request(method, self._base + path,
                                          timeout=self._timeout, **kwargs)
                if resp.status_code < 500:
                    return resp
                last = f"Search error {resp.status_code}: {resp.text}"
            except requests.RequestException as exc:
                last = f"{type(exc).__name__}: {exc}"
            if not switched and self._failover():
                switched = True
                budget = attempt + self._max_retries      # a fresh budget against the new address
                continue
            if attempt < budget:
                time.sleep(RETRY_SLEEP_S)
        message = (f"OpenResearcher search service unreachable at "
                   f"{self._base} after {self._max_retries} attempts "
                   f"({last}) — run ABORTED so no further LLM tokens are "
                   f"spent against a dead service. Restart it or set "
                   f"OR_SERVICE_URL to its current node (README.md, 4. Evaluation, Deep Research, "
                   f"openresearcher), then rerun this config: resume "
                   f"continues from checkpoints.")
        print(f"[openresearcher] {message}", flush=True)
        raise ServiceDownError(message)

    def _failover(self) -> bool:
        """Re-read the address file; switch to a NEW url if it holds one. False when there is no
        file, it is unchanged, or it does not look like an address."""
        if not self._url_file:
            return False
        try:
            with open(self._url_file) as f:
                url = f.read().strip().rstrip("/")
        except OSError:
            return False
        if not url.startswith("http://") or "\n" in url or url == self._base:
            return False
        print(f"[openresearcher] search service moved: {self._base} -> {url} (from {self._url_file}); "
              f"retrying there", flush=True)
        self._base = url
        return True

    def _probe(self) -> str | None:
        resp = self._request("GET", "/")
        if resp.status_code != 200:
            raise RuntimeError(
                f"OpenResearcher search service at {self._base} answered "
                f"{resp.status_code} to the readiness probe — is something "
                f"else on that port? (setup: README.md, 4. Evaluation, Deep Research)")
        return (resp.json() or {}).get("active_searcher")

    # Error layering for ANSWERED failures mirrors upstream's: non-200
    # becomes their BackendError text, and which words the model reads
    # depends on the calling tool — the search path shows the text raw, the
    # open path wraps it as "Error fetching URL `...`" (tools.py). Transport
    # failures deliberately DIVERGE from upstream (which fed them back as
    # tool text forever): they retry and then abort the run — see
    # ServiceDownError.

    def _search_single(self, query: str, topn: int) -> list[dict[str, Any]]:
        resp = self._request("POST", "/search",
                             json={"query": str(query), "topn": int(topn)})
        if resp.status_code != 200:
            raise BackendError(
                f"Search error {resp.status_code}: {resp.text}")
        return (resp.json() or {}).get("results", []) or []

    def search(self, query, topn: int) -> list[dict[str, Any]]:
        queries = [query] if isinstance(query, str) else [str(q) for q in query]
        if not queries:
            raise BackendError("No results returned for any query: []")
        if len(queries) == 1:
            hits = self._search_single(queries[0], topn)
        else:
            with ThreadPoolExecutor(max_workers=len(queries)) as pool:
                batches = list(pool.map(lambda q: self._search_single(q, topn),
                                        queries))
            hits = [hit for batch in batches for hit in batch]
        if not hits:
            # upstream raises when EVERY query came back empty; the tool layer
            # turns this into the text the model reads
            raise BackendError(f"No results returned for any query: {queries}")
        return hits

    def get_content(self, url: str) -> dict[str, str]:
        resp = self._request("POST", "/get_content", json={"url": str(url)})
        if resp.status_code != 200:
            raise BackendError(
                f"Search error {resp.status_code}: {resp.text}")
        payload = resp.json() or {}
        if not payload.get("content"):
            raise BackendError(f"No content returned for {url}")
        return {"title": str(payload.get("title") or ""),
                "content": str(payload.get("content") or "")}
