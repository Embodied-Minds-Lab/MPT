from collections import defaultdict

_FAIL_LIMIT = 3


class GenerationFlopsMeter:
    def __init__(self):
        self.reset()
        self._counters = {}        # model_path -> FlopsCounter
        self._fails = 0
        self._disabled = False

    def reset(self):
        self.stats = defaultdict(lambda: {"calls": 0, "seqs": 0, "prompt_tok": 0, "resp_tok": 0,
                                          "images": 0, "flops_alg": 0.0, "flops_machine": 0.0})

    # -- internals ---------------------------------------------------------------------------
    def _counter(self, model_path):
        if model_path not in self._counters:
            from verl.utils.flops_counter import FlopsCounter
            # config only: no weights, no GPU, a few ms once per worker group
            cfg = None
            try:
                from transformers import AutoConfig
                cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            except Exception:
                cfg = None
            if cfg is None or getattr(cfg, "model_type", None) is None:
                # A custom config whose auto_map points at the model's own module (NaVILA's
                # "llava_llama") may not load here without that module on the path -- and the
                # silent failure was pinning EVERY generation-FLOPs value to 0. Read config.json
                # directly instead; FlopsCounter's readers are dict-tolerant, so a plain namespace
                # (text_config / vision_config left as dicts) is enough to size the model.
                import json, os, types
                with open(os.path.join(model_path, "config.json")) as _f:
                    cfg = types.SimpleNamespace(**json.load(_f))
            fc = FlopsCounter(cfg)
            # verl's counter knows a fixed set of model types and returns 0 for the rest (it prints
            # "MFU will always be zero"): every OpenResearcher run on gemma-4 logged tokens but
            # flops=0 (09-22). Unknown text_config models get a dense estimate instead; known types
            # are untouched.
            from verl.utils.flops_counter import VALID_CONFIG_TYPE
            if getattr(cfg, "model_type", None) not in VALID_CONFIG_TYPE:
                try:
                    fc = _TextConfigDenseCounter(cfg)
                except Exception as e:  # noqa: BLE001 -- fall back to verl's zeros, never break generation
                    print(f"[flops_meter] no dense estimate for model_type={getattr(cfg, 'model_type', None)} ({e}); FLOPs stay 0")
            self._counters[model_path] = fc
        return self._counters[model_path]

    @staticmethod
    def _seq_lens(dp, resp_len):
        """Per-sequence (prompt_len, resp_len) from the attention mask.

        The mask is the only honest source: `input_ids.shape[1]` is the padded width, which would
        overcount by however much left-padding the batch happened to need.
        """
        am = dp.batch["attention_mask"]
        if resp_len:
            prompts = am[:, :-resp_len].sum(dim=-1).tolist()
            resps = am[:, -resp_len:].sum(dim=-1).tolist()
        else:
            prompts = am.sum(dim=-1).tolist()
            resps = [0] * len(prompts)
        return [int(p) for p in prompts], [int(r) for r in resps]

    @staticmethod
    def _n_images(dp):
        mm = dp.non_tensor_batch.get("multi_modal_data")
        if mm is None:
            return 0
        n = 0
        for row in mm:
            try:
                n += len(row.get("image", []) or [])
            except Exception:
                n += 1 if row else 0
        return n

    # -- the hook ----------------------------------------------------------------------------
    def observe(self, wg_id, kind, model_path, batch_out_alg, batch_out_machine, patches=256):
        """One LLM call. `batch_out_*` are the post-generation DataProtos: `_alg` after unpad
        (what the algorithm consumes), `_machine` before unpad (what the GPU actually ran)."""
        if self._disabled:
            return
        try:
            key = (str(wg_id), str(kind))
            s = self.stats[key]
            s["calls"] += 1
            resp_len = batch_out_alg.batch["responses"].shape[1]
            p_alg, r_alg = self._seq_lens(batch_out_alg, resp_len)
            img_alg = self._n_images(batch_out_alg)
            s["seqs"] += len(p_alg)
            s["prompt_tok"] += sum(p_alg)
            s["resp_tok"] += sum(r_alg)
            s["images"] += img_alg
            try:
                fc = self._counter(model_path)
                s["flops_alg"] += fc.absolute_flops([a + b for a, b in zip(p_alg, r_alg)],
                                                    "forward", n_images=img_alg,
                                                    patches_per_image=patches)
                if batch_out_machine is not None:
                    p_m, r_m = self._seq_lens(batch_out_machine, resp_len)
                    s["flops_machine"] += fc.absolute_flops(
                        [a + b for a, b in zip(p_m, r_m)], "forward",
                        n_images=self._n_images(batch_out_machine), patches_per_image=patches)
            except Exception:
                pass                      # token counts still land; FLOPs stay 0 for this call
        except Exception as e:
            self._fails += 1
            if self._fails >= _FAIL_LIMIT and not self._disabled:
                self._disabled = True
                print(f"[flops_meter] disabled after {self._fails} failures; last: {type(e).__name__}: {e}")

    # -- readout -----------------------------------------------------------------------------
    def metrics(self, prefix="gen"):
        out, tot_alg, tot_machine, tot_tok = {}, 0.0, 0.0, 0
        by_kind = defaultdict(float)
        for (wg, kind), s in self.stats.items():
            base = f"{prefix}/{wg}/{kind}"
            out[f"{base}/calls"] = s["calls"]
            out[f"{base}/tokens"] = s["prompt_tok"] + s["resp_tok"]
            out[f"{base}/flops"] = s["flops_alg"]
            tot_alg += s["flops_alg"]
            tot_machine += s["flops_machine"]
            tot_tok += s["prompt_tok"] + s["resp_tok"]
            by_kind[kind] += s["flops_alg"]
        if not self.stats:
            return out
        out[f"{prefix}/flops_total"] = tot_alg
        out[f"{prefix}/flops_machine_total"] = tot_machine
        out[f"{prefix}/tokens_total"] = tot_tok
        # how much of generation is spent on bids that are thrown away
        if tot_alg > 0:
            for kind, v in by_kind.items():
                out[f"{prefix}/share_{kind}"] = v / tot_alg
        if tot_alg > 0 and tot_machine > 0:
            out[f"{prefix}/padding_overhead"] = tot_machine / tot_alg - 1.0
        return out


class _TextConfigDenseCounter:
    """Forward-FLOPs estimate for a decoder whose sizes live in `config.text_config` (or at the top
    level) and whose model_type verl's FlopsCounter rejects -- gemma-4 (E2B/E4B) in practice.

    dense  = 2 * N * tokens, N = per-token matmul params: gated MLP (gate, up, down), q/k/v/o
             projections (K/V skipped on `num_kv_shared_layers` layers, gemma-4's KV sharing),
             the per-layer-input gate/projection (gemma-3n/4 PLE, `hidden_size_per_layer_input`),
             and the LM head once (embedding gathers cost no FLOPs).
    attn   = 4 * heads * head_dim * sum over sequences of s * ctx(s), ctx = s on full-attention
             layers and min(s, sliding_window) on sliding layers (`layer_types`).
    Same conventions as FlopsCounter.qwen3_vl_dense_params; "forward" only (generation)."""

    def __init__(self, cfg):
        t = getattr(cfg, "text_config", None) or cfg
        g = lambda k, d=None: (t.get(k, d) if isinstance(t, dict) else getattr(t, k, d))
        H, L, I, V = int(g("hidden_size")), int(g("num_hidden_layers")), int(g("intermediate_size")), int(g("vocab_size"))
        nh, nkv = int(g("num_attention_heads")), int(g("num_key_value_heads", g("num_attention_heads")))
        hd = int(g("head_dim", H // nh))
        ple = int(g("hidden_size_per_layer_input", 0) or 0)
        shared = int(g("num_kv_shared_layers", 0) or 0)
        types = list(g("layer_types", []) or [])
        self.window = int(g("sliding_window", 0) or 0)
        self.n_sliding = sum(1 for x in types if "sliding" in str(x)) if types else 0
        self.n_full = L - self.n_sliding
        q, kv, o = nh * hd, nkv * hd, nh * hd
        per_layer = H * I * 3 + H * (q + o) + 2 * H * ple
        kv_layers = L - shared
        self.N = per_layer * L + H * 2 * kv * kv_layers + V * H
        self.heads_x_dim = nh * hd
        self.model_type = g("model_type", getattr(cfg, "model_type", "?"))

    def absolute_flops(self, seqlens, pass_kind="forward", n_images=0, patches_per_image=256):
        mult = 2.0 if pass_kind == "forward" else 6.0
        tokens = sum(seqlens)
        dense = mult * self.N * tokens
        ctx = lambda s: (self.n_full * s * s + self.n_sliding * s * min(s, self.window or s))
        attn = (mult / 2.0) * 4 * self.heads_x_dim * sum(ctx(int(s)) for s in seqlens)
        return float(dense + attn)


METER = GenerationFlopsMeter()
