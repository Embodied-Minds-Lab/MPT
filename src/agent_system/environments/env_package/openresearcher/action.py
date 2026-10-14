from __future__ import annotations

import json
from typing import List, Optional, Tuple

_MAGIC = "\x1eOR1\x1e"


def encode_action(text: str, allowed: Optional[List[str]] = None, hit_cap: Optional[bool] = None) -> str:
    """`hit_cap` (09-23): whether the generation filled the whole response budget (no EOS, no padding) --
    the only honest "truncated by the length cap" signal; the env's textual heuristic misfired on 80 % of
    the turns it flagged (unclosed tags followed by a normal end of turn)."""
    if allowed is None and hit_cap is None:
        return text
    obj = {"text": text}
    if allowed is not None:
        obj["allowed"] = sorted(allowed)
    if hit_cap is not None:
        obj["hit_cap"] = bool(hit_cap)
    return _MAGIC + json.dumps(obj, ensure_ascii=False)


def _obj(action):
    if isinstance(action, str) and action.startswith(_MAGIC):
        return json.loads(action[len(_MAGIC):])
    return None


def decode_action(action: str) -> Tuple[str, Optional[set]]:
    """-> (text, allowed_tool_names or None). Never raises on a plain string."""
    obj = _obj(action)
    if obj is not None:
        return str(obj.get("text", "")), (set(obj["allowed"]) if "allowed" in obj else None)
    return (action if isinstance(action, str) else ""), None


def decode_hit_cap(action: str) -> Optional[bool]:
    """-> the generator's length-cap flag, or None when the action did not carry one (old traces, tests)."""
    obj = _obj(action)
    return None if obj is None or "hit_cap" not in obj else bool(obj["hit_cap"])
