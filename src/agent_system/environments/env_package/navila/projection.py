# Validity projection: a NaVILA response is a valid action iff it names a navigation verb
# (stop / move forward / turn left / turn right). The raw text is passed through unchanged --
# the habitat env-server parses the phrase (degree/distance) exactly like the official eval loop.
from typing import List, Tuple

from .envs import is_valid_action


def navila_projection(text_actions: List[str]) -> Tuple[List[str], List[bool]]:
    valids = [is_valid_action(t) for t in text_actions]
    return list(text_actions), valids
