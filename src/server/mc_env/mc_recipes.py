# One ladder per chain. The shared wood-and-stone prefix is identical for all three; only the two
# post-furnace steps differ, so each task is shown the items of its own goal.
_PREFIX = ("planks", "crafting_table", "stick", "wooden_pickaxe", "stone_pickaxe", "furnace")
LADDERS = {
    "iron":         _PREFIX + ("iron_ingot", "iron_pickaxe"),
    "torch":        _PREFIX + ("charcoal", "torch"),
    "stone_bricks": _PREFIX + ("stone", "stone_bricks"),
}
LADDER = LADDERS["iron"]            # the default ladder (the released task is the iron pickaxe)


def ladder_for(reward_cfg):
    """Pick the ladder from the reward identities -- the SAME test mc_env_server.rules_for uses,
    kept in step by hand because this module cannot import the server (circular)."""
    ids = set()
    for r in ([] if reward_cfg is None else reward_cfg):
        try:
            ids.add(str(dict(r).get("identity", "")))
        except Exception:
            pass
    if any(i.endswith("torch") for i in ids):
        return LADDERS["torch"]
    if any("stone_brick" in i for i in ids):
        return LADDERS["stone_bricks"]
    if any("iron" in i for i in ids):
        return LADDERS["iron"]
    return _PREFIX
_DISPLAY = {"stick": "sticks"}
_PAT = {"#log": "log", "#planks": "planks", "#fuel": "fuel (planks or coal)",
        "stick": "sticks"}


def _have(inv, pat):
    """Countable stock for a consume pattern -- same matching rule as Slot._inv_pick."""
    if pat == "#log":
        return sum(v for k, v in inv.items() if k.endswith("_log"))
    if pat == "#planks":
        return sum(v for k, v in inv.items() if k.endswith("_planks"))
    if pat == "#fuel":
        return inv.get("coal", 0) + sum(v for k, v in inv.items() if k.endswith("_planks"))
    return inv.get(pat, 0)


def craftable_line(inv, ladder=LADDER):
    """Pure function of (inventory dict, ladder) -> one 'Craftable now: ...' line (no newlines).
    `ladder` defaults to the iron ladder so every existing caller renders exactly what it did."""
    from mc_env_server import _RECIPES   # lazy: single recipe source, no import-order trap
    can, missing = [], []
    for item in ladder:
        rec = _RECIPES[item]
        short = [f"{cnt - _have(inv, pat)} {_PAT.get(pat, pat)}"
                 for pat, cnt in rec["consume"] if _have(inv, pat) < cnt]
        short += [n for n in rec["needs"] if not inv.get(n)]
        name = _DISPLAY.get(item, item)
        if not short:
            can.append(name)
        elif not inv.get(item):
            missing.append(f"{name} (need {', '.join(short)})")
    line = "Craftable now: " + (", ".join(can) or "nothing")
    if missing:
        line += "; missing: " + ", ".join(missing[:2])
    return line
