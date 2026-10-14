from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

ELIDED_TOOL = "[earlier tool response elided to fit the context window]"
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

DIALECTS = ("native", "textual")
# What an elided tool response becomes (env.openresearcher.elide_mode; must match the SFT data):
#   stub   : the one-line placeholder alone (the released deep-research agents were trained with this)
#   header : the page's first line -- `[cursor] title (url)` -- above the placeholder, so the
#            cursor space stays legible and 【cursor†L..】 citations of elided pages still name a
#            page (long episodes otherwise cite evidence that sat on elided pages)
ELIDE_MODES = ("stub", "header")
_HEADER_MAX_CHARS = 200

# Qwen3's own tool-schema wording (its chat template), reproduced as plain text for templates that
# have no `tools=` rendering. Keep the bytes stable: every SFT prompt carries them.
_TOOLS_HEAD = "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>"
_TOOLS_TAIL = '\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'


def tools_block(tools) -> str:
    """The textual-dialect tool schema, appended to the system content."""
    if not tools:
        return ""
    return _TOOLS_HEAD + "".join("\n" + json.dumps(t, ensure_ascii=False) for t in tools) + _TOOLS_TAIL


def resolve_dialect(tokenizer, value: Optional[str] = "auto") -> str:
    """Explicit `native` / `textual` wins; `auto` = native iff the chat template itself renders
    <tool_call> and <tool_response> (Qwen3). gemma-3/4 templates mention neither (gemma-4's own
    markup is <|tool_call>...<tool_call|>), so they get the textual protocol."""
    v = (value or "auto").strip().lower()
    if v in DIALECTS:
        return v
    if v != "auto":
        raise ValueError(f"tool_dialect must be one of {DIALECTS + ('auto',)}, got {value!r}")
    if tokenizer is None:
        return "native"
    tpl = getattr(tokenizer, "chat_template", None) or ""
    return "native" if ("<tool_call>" in tpl and "<tool_response>" in tpl) else "textual"


def to_template_messages(messages: List[Dict[str, Any]], tools, dialect: str) -> List[Dict[str, Any]]:
    """Env-shaped history -> what apply_chat_template receives. Always returns fresh dicts."""
    if dialect == "native":
        return [dict(m) for m in messages]
    if dialect != "textual":
        raise ValueError(f"unknown dialect {dialect!r}")
    out: List[Dict[str, Any]] = []
    block = tools_block(tools)
    seen_system = False
    for m in messages:
        role = m.get("role")
        content = m.get("content") or ""
        if role == "system":
            seen_system = True
            out.append({"role": "system", "content": content + block})
            continue
        if role == "tool":
            role, content = "user", f"<tool_response>\n{content}\n</tool_response>"
        if role == "user" and out and out[-1]["role"] == "user":
            out[-1]["content"] = out[-1]["content"] + "\n\n" + content
            continue
        out.append({"role": role, "content": content})
    if block and not seen_system:
        out.insert(0, {"role": "system", "content": block.lstrip("\n")})
    return out


def render_prompt(tokenizer, messages: List[Dict[str, Any]], tools, dialect: str, tokenize: bool = False, add_generation_prompt: bool = True, **kwargs):
    """The one call that turns a history into a prompt (text or ids). preprocess_fn, fit_messages
    and the SFT builder all go through here."""
    msgs = to_template_messages(messages, tools, dialect)
    return tokenizer.apply_chat_template(msgs, tools=(tools if dialect == "native" else None), add_generation_prompt=add_generation_prompt, tokenize=tokenize, **kwargs)


def n_prompt_tokens(tokenizer, messages: List[Dict[str, Any]], tools, dialect: str = "native") -> int:
    ids = render_prompt(tokenizer, messages, tools, dialect, tokenize=True)
    # some tokenizers return a BatchEncoding when tokenize=True (transformers 5.x always does)
    if hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    elif isinstance(ids, dict):
        ids = ids["input_ids"]
    return len(ids)


def _strip_think(msgs: List[Dict[str, Any]], i: int) -> bool:
    c = msgs[i].get("content") or ""
    nc = _THINK_RE.sub("", c, count=1).lstrip("\n")
    if nc == c:
        return False
    msgs[i]["content"] = nc
    return True


def _elide_tool(msgs: List[Dict[str, Any]], i: int, mode: str = "stub") -> bool:
    c = msgs[i].get("content") or ""
    if c == ELIDED_TOOL or c.endswith("\n" + ELIDED_TOOL):
        return False
    if mode == "header":
        head = c.split("\n", 1)[0].strip()[:_HEADER_MAX_CHARS]
        msgs[i]["content"] = f"{head}\n{ELIDED_TOOL}" if head else ELIDED_TOOL
    elif mode == "stub":
        msgs[i]["content"] = ELIDED_TOOL
    else:
        raise ValueError(f"elide_mode must be one of {ELIDE_MODES}, got {mode!r}")
    return True


def _elision_ops(messages: List[Dict[str, Any]], keep_last_k: int, elide_mode: str = "stub") -> List[Tuple[int, str]]:
    """The ordered op list (index, 'strip'|'elide'), restricted to ops that actually change the
    history when applied in order -- the linear scan skipped no-op ops without measuring, and the
    bisect must count prefixes the same way."""
    k = max(int(keep_last_k), 0)
    a_idx = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    t_idx = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    old_a, new_a = (a_idx[:-k], a_idx[-k:]) if k else (a_idx, [])
    old_t, new_t = (t_idx[:-k], t_idx[-k:]) if k else (t_idx, [])
    candidates = [(i, "strip") for i in old_a] + [(i, "elide") for i in old_t] + [(i, "elide") for i in new_t] + [(i, "strip") for i in new_a]
    work = [dict(m) for m in messages]
    ops = []
    for i, kind in candidates:
        changed = _strip_think(work, i) if kind == "strip" else _elide_tool(work, i, elide_mode)
        if changed:
            ops.append((i, kind))
    return ops


def _apply_ops(messages: List[Dict[str, Any]], ops: List[Tuple[int, str]], n: int, elide_mode: str = "stub"):
    msgs = [dict(m) for m in messages]
    stats = {"elided_tools": 0, "stripped_think": 0}
    for i, kind in ops[:n]:
        if kind == "strip":
            _strip_think(msgs, i)
            stats["stripped_think"] += 1
        else:
            _elide_tool(msgs, i, elide_mode)
            stats["elided_tools"] += 1
    return msgs, stats


def fit_messages(messages: List[Dict[str, Any]], tools, tokenizer, budget: int, keep_last_k: int = 2, dialect: str = "native", search: str = "bisect", elide_mode: str = "stub") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Returns (view, stats). `view` is a fresh list of fresh dicts; `messages` is untouched."""
    if elide_mode not in ELIDE_MODES:
        raise ValueError(f"elide_mode must be one of {ELIDE_MODES}, got {elide_mode!r}")
    msgs = [dict(m) for m in messages]
    stats: Dict[str, Any] = {"elided_tools": 0, "stripped_think": 0, "fits": True}
    if n_prompt_tokens(tokenizer, msgs, tools, dialect) <= budget:
        return msgs, stats
    ops = _elision_ops(messages, keep_last_k, elide_mode)

    def fits(n: int):
        view, st = _apply_ops(messages, ops, n, elide_mode)
        return n_prompt_tokens(tokenizer, view, tools, dialect) <= budget, view, st

    if search == "linear":
        for n in range(1, len(ops) + 1):
            ok, view, st = fits(n)
            if ok:
                return view, {**st, "fits": True}
        view, st = _apply_ops(messages, ops, len(ops), elide_mode)
    elif search == "bisect":
        ok, view, st = fits(len(ops)) if ops else (False, msgs, stats)
        if ok:
            lo, hi = 1, len(ops)
            while lo < hi:
                mid = (lo + hi) // 2
                ok_mid, view_mid, st_mid = fits(mid)
                if ok_mid:
                    hi, view, st = mid, view_mid, st_mid
                else:
                    lo = mid + 1
            return view, {**st, "fits": True}
    else:
        raise ValueError(f"search must be 'bisect' or 'linear', got {search!r}")
    print(f"[openresearcher] prompt still over budget={budget} after full elision (elided_tools={st['elided_tools']} stripped_think={st['stripped_think']}); verl data.truncation applies", flush=True)
    return view, {**st, "fits": False}
