from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

try:                       # json5 is what OpenResearcher parses tool calls with (lenient JSON)
    import json5 as _json
except ImportError:        # pragma: no cover - strict json is a usable fallback
    import json as _json

# deploy_agent.py:207 -- generation stops at the first <tool_response>; the model must never write
# its own tool result. The second string subsumes the first for stopping; both kept for fidelity.
STOP_STRINGS = ["\n<tool_response>", "<tool_response>"]

_THINK_FULL = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_THINK_TAIL = re.compile(r"^(.*?)</think>", re.DOTALL)
_TC_FULL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_TC_TAIL = re.compile(r"^(.*?)</tool_call>", re.DOTALL)
_XML_FN = re.compile(r"<function=([\w.]+)>")
_XML_PARAM = re.compile(r"<parameter=([\w]+)>\s*(.*?)\s*</parameter>", re.DOTALL)


# gemma-4's own single-token markup (special ids 48/49 tool call, 100/101 "channel" = thinking, 50/51 tool response,
# 105/106 turn). The SFT corpus teaches the TEXTUAL tags, but the base model's prior leaks the native tokens into
# 6-9 % of act rows (09-23: `native gemma markup in k/n responses`), typically as the CLOSING token of a textual
# block -- `<tool_call>{...}<tool_call|>` -- which BaseAgent's skip_special_tokens decode silently dropped, so the
# env saw an unclosed <tool_call> and ended the episode as "truncated". Decode WITH special tokens and map them.
_NATIVE_TO_TEXT = (("<|tool_call>", "<tool_call>"), ("<tool_call|>", "</tool_call>"),
                   ("<|channel>", "<think>"), ("<channel|>", "</think>"),
                   ("<|tool_response>", "<tool_response>"), ("<tool_response|>", "<tool_response>"))
_DROP_TOKENS = ("<turn|>", "<|turn>", "<eos>", "<bos>", "<pad>", "<end_of_turn>", "<start_of_turn>")


def normalize_native_markup(text: str) -> str:
    """Native gemma-4 tokens -> the textual tags the parser and the SFT corpus use; turn/eos/pad tokens removed."""
    t = text or ""
    for a, b in _NATIVE_TO_TEXT:
        t = t.replace(a, b)
    for a in _DROP_TOKENS:
        t = t.replace(a, "")
    return t


def cut_at_stop(text: str) -> str:
    """deploy_agent.py:235-237: everything from the first '<tool_response>' on is discarded."""
    if not text:
        return ""
    pos = text.find("<tool_response>")
    return text[:pos] if pos >= 0 else text


@dataclass
class ParsedTurn:
    raw: str                                   # the turn after cut_at_stop (what history stores)
    reasoning: Optional[str] = None            # <think> body, or None
    content: str = ""                          # turn with think + tool_call blocks removed
    tool_call_text: Optional[str] = None       # inner text of <tool_call>, or None
    name: Optional[str] = None                 # parsed function name (e.g. "browser.search")
    args: Dict[str, Any] = field(default_factory=dict)
    parse_error: bool = False                  # a <tool_call> block was present but unparseable
    bare_call: bool = False                    # the call came as bare `browser.x(...)` text (lenient parse)
    unclosed_tool_call: bool = False           # <tool_call> with complete JSON but no </tool_call> (lenient parse)
    answer_text: str = ""                      # OR's `non_thinking_content` for a no-tool turn

    @property
    def has_tool_call(self) -> bool:
        return self.tool_call_text is not None


_BARE_CALL = re.compile(r"(?<![\w.])(?:browser\.)?(search|open|find)\s*\(", re.IGNORECASE)
_POSITIONAL = {"search": ("query", "topn"), "open": ("id", "cursor", "loc", "num_lines"), "find": ("pattern", "cursor")}


def _lift_bare_call(content: str):
    """`browser.search(query="x", topn=5)` / `open(2)` / `find("1931")` written as plain text ->
    (json tool_call text, remaining content) or None. Arguments are parsed as a Python call
    (literals only); a call that does not parse is left alone (treated as an answer)."""
    m = _BARE_CALL.search(content or "")
    if not m:
        return None
    start, depth, i = m.end() - 1, 0, m.end() - 1
    while i < len(content):
        if content[i] == "(":
            depth += 1
        elif content[i] == ")":
            depth -= 1
            if depth == 0:
                break
        i += 1
    if i >= len(content):
        return None
    tool, argtext = m.group(1).lower(), content[start + 1:i]
    try:
        call = ast.parse(f"f({argtext})", mode="eval").body
        args: Dict[str, Any] = {}
        for k, node in zip(_POSITIONAL[tool], call.args):
            args[k] = ast.literal_eval(node)
        for kw in call.keywords:
            if kw.arg:
                args[kw.arg] = ast.literal_eval(kw.value)
    except Exception:
        return None
    rest = (content[:m.start()] + content[i + 1:]).strip()
    return _json.dumps({"name": f"browser.{tool}", "arguments": args}), rest


def parse_assistant_turn(text: str) -> ParsedTurn:
    text = cut_at_stop(text or "")
    out = ParsedTurn(raw=text)
    content = text

    # Step 1 (deploy_agent.py:239-254): extract and remove <think>...</think>
    if "<think>" in content and "</think>" in content:
        m = _THINK_FULL.search(content)
        if m:
            out.reasoning = m.group(1).strip()
            content = content.replace(m.group(0), "").strip()
    elif "</think>" in content:
        m = _THINK_TAIL.search(content)
        if m:
            out.reasoning = m.group(1).strip()
            content = content.replace(m.group(0), "").strip()

    # Step 2 (deploy_agent.py:256-273): extract and remove <tool_call>...</tool_call>
    if "<tool_call>" in content and "</tool_call>" in content:
        m = _TC_FULL.search(content)
        if m:
            out.tool_call_text = m.group(1).strip()
            content = content.replace(m.group(0), "").strip()
    elif "</tool_call>" in content:
        m = _TC_TAIL.search(content)
        if m:
            out.tool_call_text = m.group(1).strip()
            content = content.replace(m.group(0), "").strip()
    elif "<tool_call>" in content:
        # opened, never closed, but the model ended its turn right after a complete JSON object (09-23,
        # 45/56 of the chain's "truncated" turns): that IS the tool call -- take it instead of killing
        # the episode as a length-cap hit.
        tail = content[content.index("<tool_call>") + len("<tool_call>"):].strip()
        try:
            obj = _json.loads(tail)
        except Exception:
            obj = None
        if isinstance(obj, dict) and obj.get("name"):
            out.tool_call_text, out.unclosed_tool_call = tail, True
            content = content[:content.index("<tool_call>")].strip()
    out.content = content

    # Leniency (by design): the SFT agents often emit OpenResearcher's bare call syntax
    # -- `browser.search(query="...")`, `open(id=2)` -- instead of a <tool_call> block. Without this the
    # text was judged as an answer and 30-40 % of chain episodes died at turn 0. A bare call outside
    # any <tool_call> block is lifted into one (json); anything unparseable stays an answer.
    if out.tool_call_text is None:
        bare = _lift_bare_call(content)
        if bare is not None:
            out.tool_call_text, out.content, out.bare_call = bare[0], bare[1], True

    # deploy_agent.py:275-328: json5 first, XML fallback, else a parse failure
    if out.tool_call_text:
        parsed = None
        try:
            obj = _json.loads(out.tool_call_text)
            if isinstance(obj, dict):
                parsed = (str(obj.get("name", "")), obj.get("arguments", {}))
        except Exception:
            parsed = None
        if parsed is None:
            fm = _XML_FN.search(out.tool_call_text)
            if fm:
                args: Dict[str, Any] = {}
                for p in _XML_PARAM.finditer(out.tool_call_text):
                    val = p.group(2).strip()
                    if val.startswith('"') and val.endswith('"'):
                        val = val[1:-1]
                    if val.isdigit():
                        val = int(val)
                    args[p.group(1)] = val
                parsed = (fm.group(1), args)
        if parsed is None:
            out.parse_error = True
        else:
            out.name, raw_args = parsed
            if isinstance(raw_args, str):
                # deploy_agent.py:354-357: string arguments are json-decoded; failure -> error text
                try:
                    raw_args = _json.loads(raw_args)
                except Exception:
                    out.parse_error = True
                    raw_args = {}
            out.args = raw_args if isinstance(raw_args, dict) else {}
    elif out.tool_call_text == "":
        # an empty <tool_call></tool_call> block: OR treats `if tool_call_text:` as falsy and grades
        # the turn as a plain message. Match that: no tool call.
        out.tool_call_text = None

    # deploy_agent.py:334-335: the graded text of a no-tool turn is the ORIGINAL turn after </think>
    if out.tool_call_text is None:
        out.answer_text = text.split("</think>", 1)[1].strip() if "</think>" in text else text.strip()
    return out


def has_answer_marker(content: str) -> bool:
    """deploy_agent.py:401-412 -- OpenResearcher's own loop only STOPS on a no-tool turn when the
    content carries one of these markers; otherwise it keeps generating. Off by default here (a
    no-tool turn is the final answer), exposed as env.openresearcher.require_answer_marker."""
    c = (content or "").lower()
    if "<answer>" in c and "</answer>" in c:
        return True
    if "exact answer:" in c and "confidence:" in c:
        return True
    if "final answer:" in c or "answer:" in c:
        return True
    return False


def normalize_tool_name(name: Optional[str]) -> str:
    """deploy_agent.py:365-369: 'browser.search' -> 'search'; anything else passes through."""
    if not name:
        return ""
    return name.split(".", 1)[1] if name.startswith("browser.") else name
