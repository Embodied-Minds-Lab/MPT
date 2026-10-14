from __future__ import annotations

from typing import Any, Dict, Optional, Set

from .browser import (MAX_SEARCH_RESULTS, BrowserState, ToolUsageError,
                      build_fetched_page, build_find_page, build_search_page, show_page)
from .service import BackendError, ServiceClient
from .parser import normalize_tool_name

TOOL_NAMES = ("search", "open", "find")          # OpenResearcher's short names
FULL_TOOL_NAMES = ("browser.search", "browser.open", "browser.find")


def _clip(text: str, limit: int = 1024) -> str:
    """Upstream's maybe_truncate, for the error texts that embed URLs/causes."""
    return text if len(text) <= limit else text[:limit] + "…"


def _catching(fn_name: str, call):
    try:
        return call()
    except (ToolUsageError, BackendError) as exc:
        return str(exc)
    except TypeError as exc:
        return (f"Error: Invalid arguments for function '{fn_name}'. "
                f"Please check the function signature. Details: {exc}")
    except Exception as exc:                    # noqa: BLE001 — fed back, never fatal
        return (f"An unexpected error occurred while executing function "
                f"'{fn_name}': {exc}")


def _fetch(client: ServiceClient, url: str):
    try:
        doc = client.get_content(url)
    except Exception as exc:
        raise BackendError(f"Error fetching URL `{_clip(str(url))}`: {_clip(str(exc))}") from exc
    return build_fetched_page(url, doc["title"], doc["content"])


def tool_search(client: ServiceClient, state: BrowserState, query=None, topn=None) -> str:
    def run():
        if query is None:
            raise TypeError("missing required argument: 'query'")
        # topn is accepted and IGNORED -- gpt-oss's SimpleBrowserTool always asks for max_search_results
        hits = client.search(query, topn=MAX_SEARCH_RESULTS)
        page = build_search_page(query, hits)
        return show_page(state, page, loc=0)
    return _catching("search", run)


def tool_open(client: ServiceClient, state: BrowserState, id=None, cursor=None, loc=None,
              num_lines=None, view_source=None, source=None) -> str:
    del view_source, source                     # accepted for schema fidelity

    def run():
        target_id = -1 if id is None else id
        target_cursor = -1 if cursor is None else int(cursor)
        target_loc = -1 if loc is None else int(loc)
        target_num = -1 if num_lines is None else int(num_lines)
        if isinstance(target_id, str) and target_id.lstrip("-").isdigit():
            target_id = int(target_id)          # their XML fallback coerces too
        if isinstance(target_id, str):
            page = _fetch(client, target_id)    # a fully qualified URL: a direct open is a refresh
        elif target_id != -1:
            curr = state.get_page(target_cursor)
            url = curr.links.get(str(target_id))
            if url is None:
                raise ToolUsageError(f"Invalid link id `{target_id}`.")
            page = state.get_page_by_url(url) or _fetch(client, url)
        else:
            page = state.get_page(target_cursor)  # bare open(): re-push the cursor's page
        return show_page(state, page, loc=target_loc, num_lines=target_num)
    return _catching("open", run)


def tool_find(client: ServiceClient, state: BrowserState, pattern=None, cursor=None) -> str:
    def run():
        if pattern is None:
            raise TypeError("missing required argument: 'pattern'")
        source = state.get_page(-1 if cursor is None else int(cursor))
        page = build_find_page(str(pattern), source)
        return show_page(state, page, loc=0)
    return _catching("find", run)


_DISPATCH = {"search": tool_search, "open": tool_open, "find": tool_find}


def execute_tool(name: Optional[str], args: Dict[str, Any], state: BrowserState,
                 client: ServiceClient, allowed: Optional[Set[str]] = None) -> str:
    """Run one parsed tool call. `allowed` (full names, e.g. {'browser.search'}) is the calling
    agent's whitelist; None = everything. An off-whitelist or unknown tool gets OpenResearcher's
    own message (deploy_agent.py:376), so the model reads exactly what their harness would say.

    Returns the tool response text. Raises only ServiceDownError (BaseException)."""
    short = normalize_tool_name(name).lower()
    if short not in _DISPATCH or (allowed is not None and (name or "") not in allowed
                                  and f"browser.{short}" not in allowed):
        return f"Tool {name} not available"
    fn = _DISPATCH[short]
    kwargs = dict(args or {})
    try:
        return fn(client, state, **kwargs)
    except TypeError as exc:
        # unexpected keyword names in the model's arguments land here (before the tool body runs)
        return (f"Error: Invalid arguments for function '{short}'. "
                f"Please check the function signature. Details: {exc}")


def is_tool_available(name: Optional[str]) -> bool:
    return normalize_tool_name(name).lower() in _DISPATCH
