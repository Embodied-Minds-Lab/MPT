# Validity projection, kept for the EnvironmentManagerBase interface. The MPT env managers do not
# call it (the agents stamp is_action_valid themselves). A turn is valid iff it parses: either a well-formed
# <tool_call> or a plain (answer) turn. The action text itself is passed through untouched --
# the env appends exactly what the model wrote to the conversation.
from typing import List, Tuple

from .action import decode_action
from .parser import parse_assistant_turn


def openresearcher_projection(text_actions: List[str]) -> Tuple[List[str], List[bool]]:
    valids = []
    for t in text_actions:
        text, _ = decode_action(t)
        valids.append(not parse_assistant_turn(text).parse_error)
    return list(text_actions), valids
