from __future__ import annotations

_REPORTED: dict = {}


def trim_response_mask(batch, pad_token_id, tag: str = "") -> int:
    """Zero the attention mask on the right-padding of every response; returns the number of mask
    entries cleared. No-op when the tokenizer has no pad id (sglang then pads with EOS and the
    EOS-based mask is already exact)."""
    if pad_token_id is None:
        return 0
    resp = batch.batch.get("responses", None)
    am = batch.batch.get("attention_mask", None)
    if resp is None or am is None or resp.dim() != 2:
        return 0
    L = resp.shape[1]
    keep = (resp != pad_token_id).to(am.dtype).cumprod(dim=-1)   # 1 until the first pad, 0 after
    before = int(am[:, -L:].sum().item())
    am[:, -L:] = am[:, -L:] * keep
    cleared = before - int(am[:, -L:].sum().item())
    if cleared and _REPORTED.get(tag, 0) < 2:                     # two lines per (process, kind), then quiet
        _REPORTED[tag] = _REPORTED.get(tag, 0) + 1
        rows = int((keep[:, -1] == 0).sum().item())
        print(f"[openresearcher] {tag}: response mask trimmed on {rows}/{resp.shape[0]} rows "
              f"({cleared} pad positions; stop-string endings carry no EOS)", flush=True)
    return cleared
