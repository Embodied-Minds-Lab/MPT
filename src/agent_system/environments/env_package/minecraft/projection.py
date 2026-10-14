# Validity projection: an action is valid iff it carries a non-empty <action>...</action>.
from typing import List, Tuple
from .envs import extract_instruction


def minecraft_projection(text_actions: List[str]) -> Tuple[List[str], List[bool]]:
    valids = [extract_instruction(t) is not None for t in text_actions]
    return list(text_actions), valids
