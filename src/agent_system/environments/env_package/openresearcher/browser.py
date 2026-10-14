from __future__ import annotations

import re
import textwrap
import time
from dataclasses import dataclass, field
from urllib.parse import unquote, urlsplit

# A native reimplementation of the browser OpenResearcher's agents used —
# gpt-oss's SimpleBrowserTool (Apache-2.0), which cannot be imported here (the
# gpt-oss/openai-harmony packages are deliberately not installed, and their
# tool operates on async harmony Message objects). The semantics below follow
# that source, read as the reference spec:
#
#   * a per-agent PAGE STACK; every successful search/open/find pushes a page
#     and the cursor is the stack index (bare `open()` re-pushes the current
#     page, so the cursor still advances — that is upstream's behavior);
#   * viewports: text wrapped at WRAP_WIDTH columns, `L{n}: ` line numbers,
#     window sized to VIEW_TOKENS tokens of o200k_base, header
#     `[{cursor}] {title} ({url})` + `**viewing lines [a - b] of N**`;
#   * links rendered as 【{id}†{text}】 / 【{id}†{text}†{domain}】 and opened
#     by id; `find` returns its own result page of `# 【{idx}†match at L{n}】`
#     entries; citations are `【{cursor}†L{x}(-L{y})?】` and are never
#     post-processed;
#   * errors are MESSAGES, not exceptions, at the tool boundary — every
#     user-facing error string here matches upstream verbatim.
#
# One declared divergence (recorded in the adapter docstring): upstream renders
# pages through lxml+html2text (process_html). This corpus's documents are
# markdown-ish text and the search page is synthetic, so pages here are built
# directly from text.
#
# The page BYTES are pinned to the OpenResearcher-Dataset trajectories: the SFT
# data carries the dataset's tool pages verbatim, so every glyph this renderer emits must
# match what those pages look like, or the SFT checkpoint meets a different
# observation distribution. Pinned against the dataset:
#   search page : `[c] <query> (web-search://ts=<epoch>)`, body `L0:` blank,
#                 `L1: URL: web-search://ts=…`, `L2: # Search Results`, `L3:`
#                 blank, then one `  * 【id†title†domain】 summary` per hit
#                 (TWO leading spaces; the URL is a timestamp, never the query)
#   find page   : url `<source.url>/find?pattern=<pattern>`, body starts at
#                 `L0: # 【0†match at L<n>】` (no blank / URL lines), entries
#                 separated by one blank line; no match -> the single line
#                 `No \`find\` results for pattern: \`<pattern>\``
#   open page   : `[c] <title> (<url>)`, `L0:` blank, `L1: URL: <url>`, text
# These were checked against literal dataset excerpts.

WRAP_WIDTH = 80
VIEW_TOKENS = 1024
# gpt-oss SimpleBrowserTool(max_search_results=20): the model's `topn` argument
# is accepted and DISCARDED — the backend always receives 20
MAX_SEARCH_RESULTS = 20
FIND_MAX_RESULTS = 50
FIND_SHOW_LINES = 4
URL_MAX_CHARS = 1024        # upstream maybe_truncate() on the header's URL


class ToolUsageError(Exception):
    """A model mistake (bad cursor, bad link id, bad loc) — rendered back to
    the model as tool text, never raised past the tool boundary."""


def _count_tokens_tiktoken(text: str):
    import tiktoken
    return len(tiktoken.get_encoding("o200k_base").encode(text))


_token_counter = None
_token_counter_name = ""


def count_tokens(text: str) -> int:
    """o200k_base tokens (upstream's viewport measure), with a 4-chars/token
    fallback so an offline machine (tiktoken fetches its BPE file on first
    use) still browses — the viewport is a display budget, not a metric.
    Which counter is active is printed ONCE: the fallback changes how many
    lines a viewport shows, i.e. the observation distribution, and it used to
    engage silently (set TIKTOKEN_CACHE_DIR to a warmed cache on offline nodes)."""
    global _token_counter, _token_counter_name
    if _token_counter is None:
        try:
            _count_tokens_tiktoken("probe")
            _token_counter = _count_tokens_tiktoken
            _token_counter_name = "tiktoken o200k_base"
        except Exception as exc:                # noqa: BLE001 — offline fallback
            _token_counter = lambda text: max(1, len(text) // 4)  # noqa: E731
            _token_counter_name = f"4-chars/token FALLBACK ({type(exc).__name__}: {str(exc)[:80]})"
        print(f"[openresearcher-browser] viewport token counter: {_token_counter_name}", flush=True)
    return _token_counter(text)


def escape_special(text: str) -> str:
    """Upstream _replace_special_chars: raw 【/】 in content become 〖/〗 so
    they can never be confused with real link markup."""
    return str(text).replace("【", "〖").replace("】", "〗")


def link_markup(link_id: int | str, text: str, domain: str | None = None) -> str:
    """【id†text】 (same-domain) or 【id†text†domain】 (cross-domain); † inside
    the text becomes ‡ (upstream's rewrite, so markup stays parseable)."""
    text = escape_special(text).replace("†", "‡")
    if domain:
        return f"【{link_id}†{text}†{domain}】"
    return f"【{link_id}†{text}】"


def domain_of(url: str) -> str:
    netloc = urlsplit(url).netloc
    return netloc or url


@dataclass
class Page:
    url: str
    title: str
    text: str                                   # full body, unwrapped
    links: dict[str, str] = field(default_factory=dict)   # link id -> url

    def wrapped_lines(self) -> list[str]:
        lines: list[str] = []
        for raw in self.text.split("\n"):
            lines.extend(textwrap.wrap(raw, width=WRAP_WIDTH) or [""])
        return lines


class BrowserState:
    """One agent's browser: page cache + page stack. The cursor shown in every
    viewport header is the page's index on THIS stack, so cursors (and the
    citations built from them) are meaningful only to their own agent."""

    def __init__(self):
        self.pages: dict[str, Page] = {}
        self.page_stack: list[str] = []

    @property
    def cursor(self) -> int:
        return len(self.page_stack) - 1

    def add_page(self, page: Page) -> None:
        self.pages[page.url] = page
        self.page_stack.append(page.url)

    def get_page(self, cursor: int = -1) -> Page:
        if not self.page_stack:
            raise ToolUsageError(
                f"Cursor `{cursor}` is out of range. "
                f"Available cursor indices: [0 - {self.cursor}].")
        if cursor == -1 or cursor == self.cursor:
            return self.pages[self.page_stack[-1]]
        try:
            return self.pages[self.page_stack[cursor]]
        except IndexError:
            raise ToolUsageError(
                f"Cursor `{cursor}` is out of range. "
                f"Available cursor indices: [0 - {self.cursor}].") from None

    def get_page_by_url(self, url: str) -> Page | None:
        return self.pages.get(url)

    def pop_page(self) -> None:
        if self.page_stack:
            self.page_stack.pop()

    # ---- checkpoint support --------------------------------------------------
    def export_state(self) -> dict:
        """Plain-JSON snapshot. The page stack IS the cursor space: without it,
        a resumed agent's earlier 【cursor†L..】 citations and `open(cursor=..)`
        calls would point at pages a fresh browser never saw."""
        return {"pages": {url: {"url": p.url, "title": p.title, "text": p.text,
                                "links": dict(p.links)}
                          for url, p in self.pages.items()},
                "page_stack": list(self.page_stack)}

    def restore_state(self, state: dict) -> None:
        """Mutates IN PLACE — the tools were handed this object when the
        agents were built (the framework's restore contract)."""
        self.pages.clear()
        for url, p in (state.get("pages") or {}).items():
            self.pages[url] = Page(url=p["url"], title=p["title"],
                                   text=p["text"], links=dict(p.get("links") or {}))
        self.page_stack.clear()
        self.page_stack.extend(state.get("page_stack") or [])


# ---- page builders -----------------------------------------------------------

def build_search_page(query, hits: list[dict], ts: int | None = None) -> Page:
    """The synthetic search-results page in the OpenResearcher-Dataset's own
    shape (see the module header): title = the query, url = a timestamp
    (``ts``; wall clock when None -- the dataset's pages carry the epoch second
    of the search, never the query text), hits as ``  * 【id†title†domain】
    summary``. ``query`` may be a string or a list of strings — OpenResearcher's
    backend extension; a list's hits arrive concatenated."""
    queries = [query] if isinstance(query, str) else [str(q) for q in query]
    title = " | ".join(queries)
    url = f"web-search://ts={int(time.time()) if ts is None else int(ts)}"
    lines = [f"\nURL: {url}", "# Search Results", ""]
    links: dict[str, str] = {}
    for i, hit in enumerate(hits):
        hit_url = str(hit.get("url") or "")
        hit_title = str(hit.get("title") or hit_url)
        summary = escape_special(str(hit.get("summary") or ""))
        links[str(i)] = hit_url
        lines.append(f"  * {link_markup(i, hit_title, domain_of(hit_url))} {summary}")
    return Page(url=url, title=title, text="\n".join(lines), links=links)


def build_fetched_page(url: str, title: str, content: str) -> Page:
    """A corpus document as a page. Upstream's display_urls=True prepends the
    URL line; the corpus text is markdown-ish and carries no <a> links, so the
    page has no link registry."""
    text = f"\nURL: {url}\n{escape_special(content)}"
    return Page(url=url, title=str(title or url), text=text, links={})


_LINK_STRIP = re.compile(r"【\d+†([^†】]+)(?:†[^†】]+)?】")


def build_find_page(pattern: str, source: Page) -> Page:
    """`find` results as their own page (upstream run_find_in_page): lowercase
    substring match over the source's wrapped, link-stripped lines; at most
    FIND_MAX_RESULTS matches, each shown with FIND_SHOW_LINES lines of context
    and a link back to the source page."""
    needle = pattern.lower()
    lines = source.wrapped_lines()
    stripped = [_LINK_STRIP.sub(r"\1", line) for line in lines]
    entries = []
    links: dict[str, str] = {}
    for idx, line in enumerate(stripped):
        if needle in line.lower():
            snippet = "\n".join(stripped[idx:idx + FIND_SHOW_LINES])
            links[str(len(entries))] = source.url
            entries.append(f"# 【{len(entries)}†match at L{idx}】\n{snippet}")
            if len(entries) >= FIND_MAX_RESULTS:
                break
    title = f"Find results for text: `{pattern}` in `{source.title}`"
    # dataset shape: the find page's url is the source url + `/find?pattern=`,
    # and its body starts at the first entry -- no blank line, no `URL:` line
    url = f"{source.url}/find?pattern={pattern}"
    if not entries:
        return Page(url=url, title=title, text=f"No `find` results for pattern: `{pattern}`", links={})
    return Page(url=url, title=title, text="\n\n".join(entries), links=links)


# ---- viewport rendering ------------------------------------------------------

def _end_loc(lines: list[str], loc: int, num_lines: int) -> int:
    total = len(lines)
    if num_lines and num_lines > 0:
        return min(loc + num_lines, total)
    end, used = loc, 0
    while end < total:
        cost = count_tokens(f"L{end}: {lines[end]}\n")
        if end > loc and used + cost > VIEW_TOKENS:
            break
        used += cost
        end += 1
    return max(end, loc + 1)


def render_viewport(state: BrowserState, page: Page,
                    loc: int = -1, num_lines: int = -1) -> str:
    """The tool response for the page at the TOP of the stack — upstream's
    _render_browsing_display shape, byte-pinned by test_browser.py."""
    lines = page.wrapped_lines()
    total = len(lines)
    loc = 0 if loc is None or loc < 0 else int(loc)
    if loc >= total:
        raise ToolUsageError(
            f"Invalid location parameter: `{loc}`. "
            f"Cannot exceed page maximum of {total - 1}.")
    end = _end_loc(lines, loc, num_lines)
    body = "\n".join(f"L{i + loc}: {line}"
                     for i, line in enumerate(lines[loc:end]))
    domain = unquote(page.url)
    if len(domain) > URL_MAX_CHARS:
        domain = domain[:URL_MAX_CHARS] + "…"
    header = (f"{page.title} ({domain})\n"
              f"**viewing lines [{loc} - {end - 1}] of {total - 1}**\n\n")
    return f"[{state.cursor}] {header}{body}"


def show_page(state: BrowserState, page: Page,
              loc: int = -1, num_lines: int = -1) -> str:
    """Push, then render; a rendering error UN-pushes (upstream's
    show_page_safely pops the stack on ToolUsageError) so a failed view never
    burns a cursor."""
    state.add_page(page)
    try:
        return render_viewport(state, page, loc=loc, num_lines=num_lines)
    except ToolUsageError:
        state.pop_page()
        raise
