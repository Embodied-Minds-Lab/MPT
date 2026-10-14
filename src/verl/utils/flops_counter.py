# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import torch
from transformers import PretrainedConfig
from verl.utils.device import is_cuda_available, get_torch_device

VALID_CONFIG_TYPE = {"llama", "qwen2", "qwen2_vl", "qwen2_5_vl", "qwen3", "qwen3_vl", "qwen3_moe", "deepseek_v3", "llava_llama"}


def get_device_flops(unit="T"):
    """This device's achievable dense BF16 GEMM throughput, used as the MFU denominator.

    The Blackwell entries below are MEASURED (square bf16 matmul, best of
    n=4096/8192/16384), not read off a spec sheet: spec numbers fold in sparsity and a different
    precision, and the achievable number is what MFU should be measured against.

        NVIDIA B200                              1649 TFLOP/s
        NVIDIA RTX PRO 6000 Blackwell             402 TFLOP/s

    Why this mattered: any unrecognised device fell through to `float("inf")`, which silently
    made `mfu = achieved / inf = 0`. Combined with `qwen3_vl` being absent from the estimator
    dispatch (which made the numerator 0 too), MFU was zero for two independent reasons -- so
    fixing either one alone would still have shown 0 and taught us nothing.
    """

    def unit_convert(number, level):
        units = ["B", "K", "M", "G", "T", "P"]
        if number <= 0:
            return number
        ptr = 0
        while ptr < len(units) and units[ptr] != level:
            number /= 1000
            ptr += 1
        return number

    override = os.environ.get("VERL_DEVICE_PEAK_TFLOPS")
    if override:
        return unit_convert(float(override) * 1e12, unit)

    try:
        device_name = get_torch_device().get_device_name()
    except Exception:
        # No accelerator (e.g. re-analysing a finished run on a login node): torch.cpu has no
        # get_device_name and this used to raise, which made the counter unusable offline.
        return unit_convert(float("inf"), unit)
    flops = float("inf")  # INF flops for unkown gpu type

    if "MI300X" in device_name:
        flops = 1336e12
    elif "B200" in device_name:
        flops = 1649e12          # measured on this cluster
    elif "RTX PRO 6000" in device_name:
        flops = 402e12           # measured on this cluster
    elif "H100" in device_name or "H800" in device_name:
        flops = 989e12
    elif "A100" in device_name or "A800" in device_name:
        flops = 312e12
    elif "L40" in device_name:
        flops = 181.05e12
    elif "L20" in device_name:
        flops = 119.5e12
    elif "H20" in device_name:
        flops = 148e12
    elif "910B" in device_name:
        flops = 354e12
    else:
        print(f"[flops_counter] unknown device {device_name!r}: MFU will be 0. "
              f"Set VERL_DEVICE_PEAK_TFLOPS to fix.")
    flops_unit = unit_convert(flops, unit)
    return flops_unit


class FlopsCounter:
    """
    Used to count mfu during training loop

    Example:
        flops_counter = FlopsCounter(config)
        flops_achieved, flops_promised = flops_counter.estimate_flops(tokens_list, delta_time)

    """

    def __init__(self, config: PretrainedConfig):
        if config.model_type not in VALID_CONFIG_TYPE:
            print(f"Only support config type of {VALID_CONFIG_TYPE}, but got {config.model_type}. MFU will always be zero.")

        self.estimate_func = {
            "qwen2": self._estimate_qwen2_flops,
            "llama": self._estimate_qwen2_flops,
            "qwen2_vl": self._estimate_qwen2_flops,
            "qwen2_5_vl": self._estimate_qwen2_flops,
            "qwen3": self._estimate_qwen2_flops,
            "qwen3_vl": self._estimate_qwen3_vl_flops,
            "qwen3_moe": self._estimate_qwen3_moe_flops,
            "deepseek_v3": self._estimate_deepseek_v3_flops,
            # NaVILA (VILA-llama3-8B): the LLM lives under config.text_config (a nested LLaMA,
            # exactly like qwen3_vl) and the SigLIP tower under config.vision_config.
            "llava_llama": self._estimate_llava_llama_flops,
        }
        self.config = config

    def _estimate_unknown_flops(self, tokens_sum, batch_seqlens, delta_time):
        return 0

    def _estimate_qwen2_flops(self, tokens_sum, batch_seqlens, delta_time):
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        num_hidden_layers = self.config.num_hidden_layers
        num_key_value_heads = self.config.num_key_value_heads
        num_attention_heads = self.config.num_attention_heads
        intermediate_size = self.config.intermediate_size

        head_dim = getattr(self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads)
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        # non-attn per layer parm
        # Qwen2/LLama use SwiGelu, gate, having up and down linear layer in mlp
        mlp_N = hidden_size * intermediate_size * 3
        attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
        emd_and_lm_head_N = vocab_size * hidden_size * 2
        # non-attn all_layer parm
        dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
        # non-attn all_layer & all_token fwd & bwd flops
        dense_N_flops = 6 * dense_N * tokens_sum

        # attn all_layer & all_token fwd & bwd flops
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen
        attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

        # all_layer & all_token fwd & bwd flops
        flops_all_token = dense_N_flops + attn_qkv_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    # ---------------------------------------------------------------- Qwen3-VL ----------------
    def qwen3_vl_dense_params(self):
        """Per-token dense parameter count of the TEXT stack (attention matmuls excluded).

        Qwen3-VL keeps nothing at the top level of its config -- hidden_size, num_hidden_layers
        and friends all live under `config.text_config`. That is why simply adding "qwen3_vl" to
        the dispatch dict and reusing `_estimate_qwen2_flops` would raise AttributeError instead
        of working: that function reads `self.config.hidden_size` directly.

        One deliberate departure from the other estimators here: they charge
        `vocab * hidden * 2` for "embedding + lm_head". The input embedding is a GATHER, not a
        matmul -- it costs no FLOPs (and its backward is a scatter-add, also not a matmul). Only
        the LM head is a real `vocab x hidden` projection. We charge it once. With
        tie_word_embeddings=True the two share weights, which changes memory, not compute.
        """
        t = self.config.text_config
        head_dim = getattr(t, "head_dim", t.hidden_size // t.num_attention_heads)
        q = t.num_attention_heads * head_dim
        kv = t.num_key_value_heads * head_dim
        mlp_N = t.hidden_size * t.intermediate_size * 3          # SwiGLU: gate, up, down
        attn_linear_N = t.hidden_size * (q + kv + kv + t.num_attention_heads * head_dim)
        lm_head_N = t.vocab_size * t.hidden_size                 # see docstring: charged once
        return (mlp_N + attn_linear_N) * t.num_hidden_layers + lm_head_N

    def qwen3_vl_vision_flops(self, n_images, patches_per_image):
        """Forward FLOPs of the vision tower for `n_images` images.

        Not optional bookkeeping: at 128x128 the processor upscales to meet its min-pixels floor
        and emits image_grid_thw [1,16,16] = 256 patches -> 64 merged tokens per screenshot, and
        EVERY LLM call in a turn re-encodes the frame. For a 3-bid + 1-act turn that is four
        tower passes, a real cost of the agent society.
        """
        v = getattr(self.config, "vision_config", None)
        if v is None or not n_images:
            return 0.0
        h, i, L = v.hidden_size, v.intermediate_size, v.depth
        per_layer_N = 4 * h * h + 2 * h * i        # qkv+o ; ViT MLP is up+down (no gate)
        dense = 2 * per_layer_N * L * patches_per_image
        attn = 4 * (patches_per_image ** 2) * h * L
        merge = getattr(v, "spatial_merge_size", 2) ** 2
        merger = 2 * (h * merge) * getattr(v, "out_hidden_size", h) * (patches_per_image // merge)
        return (dense + attn + merger) * n_images

    # ---------------------------------------------------------------- LLaVA / NaVILA -----------
    @staticmethod
    def _cfg_get(cfg, key, default=None):
        """text_config / vision_config may deserialize to a nested PretrainedConfig OR a raw dict
        (the custom NaVILAConfig does not always promote them) -- read either shape."""
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    def _llava_text_cfg(self):
        return getattr(self.config, "text_config", None) or getattr(self.config, "llm_cfg", None)

    def llava_dense_params(self):
        """Per-token dense param count of NaVILA's LLaMA text stack (attn matmuls excluded).
        Mirrors qwen3_vl_dense_params but reads a dict-or-object text_config; lm_head charged once
        (the input embedding is a gather, not a matmul)."""
        t = self._llava_text_cfg()
        g = self._cfg_get
        hidden = g(t, "hidden_size"); heads = g(t, "num_attention_heads")
        kv = g(t, "num_key_value_heads", heads); inter = g(t, "intermediate_size")
        vocab = g(t, "vocab_size"); layers = g(t, "num_hidden_layers")
        head_dim = g(t, "head_dim", hidden // heads)
        q = heads * head_dim; kvd = kv * head_dim
        mlp_N = hidden * inter * 3                                # SwiGLU: gate, up, down
        attn_linear_N = hidden * (q + kvd + kvd + heads * head_dim)
        lm_head_N = vocab * hidden                               # head once; embedding is free
        return (mlp_N + attn_linear_N) * layers + lm_head_N

    def llava_vision_flops(self, n_images, patches_per_image=None):
        """Forward FLOPs of the SigLIP vision tower for `n_images` images. SigLIP is a plain ViT
        (GELU MLP = up+down, no gate, no spatial merge) with `num_hidden_layers` layers, so it
        cannot reuse qwen3_vl_vision_flops (which reads `.depth` and a spatial merger)."""
        v = getattr(self.config, "vision_config", None)
        if v is None or not n_images:
            return 0.0
        g = self._cfg_get
        h = g(v, "hidden_size"); i = g(v, "intermediate_size")
        L = g(v, "num_hidden_layers", g(v, "depth", 0))
        if patches_per_image is None:
            ps = g(v, "patch_size", 14); isz = g(v, "image_size", 384)
            patches_per_image = (isz // ps) ** 2                 # SigLIP-384/14 -> 27^2 = 729
        per_layer_N = 4 * h * h + 2 * h * i                      # qkv+o ; ViT MLP up+down (no gate)
        dense = 2 * per_layer_N * L * patches_per_image
        attn = 4 * (patches_per_image ** 2) * h * L
        return (dense + attn) * n_images

    def _estimate_llava_llama_flops(self, tokens_sum, batch_seqlens, delta_time):
        try:
            t = self._llava_text_cfg()
            head_dim = self._cfg_get(t, "head_dim",
                                     self._cfg_get(t, "hidden_size") // self._cfg_get(t, "num_attention_heads"))
            dense_N_flops = 6 * self.llava_dense_params() * tokens_sum
            seqlen_square_sum = sum(s * s for s in batch_seqlens)
            attn_qkv_flops = 12 * seqlen_square_sum * head_dim \
                * self._cfg_get(t, "num_attention_heads") * self._cfg_get(t, "num_hidden_layers")
            return (dense_N_flops + attn_qkv_flops) * (1.0 / delta_time) / 1e12
        except Exception as e:                                   # never crash training over MFU
            print(f"[flops] llava_llama estimate failed ({e}); returning 0")
            return 0

    def _estimate_qwen3_vl_flops(self, tokens_sum, batch_seqlens, delta_time):
        t = self.config.text_config
        head_dim = getattr(t, "head_dim", t.hidden_size // t.num_attention_heads)
        dense_N_flops = 6 * self.qwen3_vl_dense_params() * tokens_sum
        seqlen_square_sum = sum(s * s for s in batch_seqlens)
        attn_qkv_flops = 12 * seqlen_square_sum * head_dim * t.num_attention_heads * t.num_hidden_layers
        return (dense_N_flops + attn_qkv_flops) * (1.0 / delta_time) / 1e12

    def _estimate_deepseek_v3_flops(self, tokens_sum, batch_seqlens, delta_time):
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        moe_intermediate_size = self.config.moe_intermediate_size
        num_hidden_layers = self.config.num_hidden_layers
        first_k_dense_replace = self.config.first_k_dense_replace
        num_query_heads = self.config.num_attention_heads
        moe_num_expert = self.config.n_routed_experts

        moe_topk = self.config.num_experts_per_tok
        share_expert_num = self.config.n_shared_experts

        # non-attn per layer parm
        moe_gata_N = hidden_size * moe_num_expert
        # moe has fc1_1, fc1_2 and fc2 using SwiGLU in ExpertMlp layer & shared experts
        moe_expertmlp_N = hidden_size * moe_intermediate_size * (moe_topk + share_expert_num) * 3
        # MLA attn
        attn_linear_N = 0
        q_head_dim = self.config.qk_nope_head_dim + self.config.qk_rope_head_dim
        if self.config.q_lora_rank is None:
            attn_linear_N += hidden_size * num_query_heads * q_head_dim
        else:
            attn_linear_N += hidden_size * self.config.q_lora_rank
            attn_linear_N += num_query_heads * q_head_dim * self.config.q_lora_rank

        attn_linear_N += hidden_size * (self.config.kv_lora_rank + self.config.qk_rope_head_dim)
        attn_linear_N += num_query_heads * (q_head_dim - self.config.qk_rope_head_dim + self.config.v_head_dim) * self.config.kv_lora_rank
        attn_linear_N += num_query_heads * self.config.v_head_dim * hidden_size
        emd_and_lm_head_N = vocab_size * hidden_size * 2
        # non-attn all_layer parm
        moe_N = (moe_gata_N + moe_expertmlp_N + attn_linear_N) * (num_hidden_layers - first_k_dense_replace) + (hidden_size * self.config.intermediate_size * 3 + attn_linear_N) * first_k_dense_replace + emd_and_lm_head_N
        # non-attn all_layer & all_token fwd & bwd flops
        dense_N_flops = 6 * moe_N * tokens_sum

        # attn all_layer & all_token fwd & bwd flops
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen * num_hidden_layers

        attn_qkv_flops = 12 * seqlen_square_sum * q_head_dim * num_query_heads
        # all_layer & all_token fwd & bwk flops
        flops_all_token = dense_N_flops + attn_qkv_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12

        return flops_achieved

    def _estimate_qwen3_moe_flops(self, tokens_sum, batch_seqlens, delta_time):
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        num_hidden_layers = self.config.num_hidden_layers
        num_key_value_heads = self.config.num_key_value_heads
        num_attention_heads = self.config.num_attention_heads
        moe_intermediate_size = self.config.moe_intermediate_size       
        moe_topk = self.config.num_experts_per_tok
        num_experts = self.config.num_experts

        head_dim = getattr(self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads)
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        # non-attn per layer parm
        # gate + moe export
        moe_mlp_N = hidden_size * moe_topk * moe_intermediate_size * 3 + hidden_size * num_experts
        attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
        emd_and_lm_head_N = vocab_size * hidden_size * 2
        # non-attn all_layer parm
        dense_N = (moe_mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
        # non-attn all_layer & all_token fwd & bwd flops
        dense_N_flops = 6 * dense_N * tokens_sum

        # attn all_layer & all_token fwd & bwd flops
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen
        attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

        # all_layer & all_token fwd & bwd flops
        flops_all_token = dense_N_flops + attn_qkv_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved


    # 3 = 1 fwd + 2 bwd; +1 more fwd when activations are recomputed. Multiply by 2 FLOP/MAC.
    PASS_MULT = {"forward": 2, "train": 6, "train_ckpt": 8}

    def absolute_flops(self, batch_seqlens, pass_kind="train", n_images=0, patches_per_image=256):
        """Raw FLOPs for one pass over `batch_seqlens`. No time, no ratio, no unit scaling.

        This is the number worth reporting. `estimate_flops` below only ever gave a
        FLOP/s figure and a device peak, so the quantity you actually want -- "how much compute
        did this experiment cost" -- was never available, even before the two bugs that pinned
        MFU to zero.

        `pass_kind`:
          forward     generation and log-prob passes            2N per token
          train       fwd+bwd without activation checkpointing  6N
          train_ckpt  fwd+bwd with checkpointing (one extra fwd) 8N
        Gradient checkpointing is ON in this project's SFT and RL configs, so `train_ckpt` is the
        honest multiplier for the actor update; using 6N understates it by a third.

        Counting generation with this (pass_kind="forward", one call per LLM invocation) makes the
        society's compute explicit: a turn issues three bid calls plus one action call, and those
        bid forwards -- plus their vision-tower passes -- are real compute that GPU-hours would
        attribute to "the method being slow" rather than to the method doing more work.
        """
        mult = self.PASS_MULT[pass_kind]
        tokens_sum = sum(batch_seqlens)
        if self.config.model_type == "qwen3_vl":
            dense = mult * self.qwen3_vl_dense_params() * tokens_sum
            t = self.config.text_config
            head_dim = getattr(t, "head_dim", t.hidden_size // t.num_attention_heads)
            # attention is 2 matmuls (QK^T and AV); per token-pair cost scales with seqlen^2
            attn = (mult / 3.0) * 6 * sum(s * s for s in batch_seqlens) * head_dim \
                * t.num_attention_heads * t.num_hidden_layers
            vis = self.qwen3_vl_vision_flops(n_images, patches_per_image)
            if pass_kind != "forward":
                vis *= mult / 2.0
            return dense + attn + vis
        if self.config.model_type == "llava_llama":
            # NaVILA: LLaMA text stack (text_config) + SigLIP tower (vision_config); same shape as
            # the qwen3_vl branch above but with the LLaVA/SigLIP-specific readers. Fails safe to 0.
            try:
                t = self._llava_text_cfg()
                head_dim = self._cfg_get(t, "head_dim",
                                         self._cfg_get(t, "hidden_size") // self._cfg_get(t, "num_attention_heads"))
                dense = mult * self.llava_dense_params() * tokens_sum
                attn = (mult / 3.0) * 6 * sum(s * s for s in batch_seqlens) * head_dim \
                    * self._cfg_get(t, "num_attention_heads") * self._cfg_get(t, "num_hidden_layers")
                vis = self.llava_vision_flops(n_images)
                if pass_kind != "forward":
                    vis *= mult / 2.0
                return dense + attn + vis
            except Exception as e:
                print(f"[flops] llava_llama absolute_flops failed ({e}); returning 0")
                return 0
        # other model types: reuse the per-second estimator with delta_time=1 and rescale
        per_sec_tflops = self.estimate_func.get(
            self.config.model_type, self._estimate_unknown_flops)(tokens_sum, batch_seqlens, 1.0)
        return per_sec_tflops * 1e12 * mult / 6.0

    def estimate_flops(self, batch_seqlens, delta_time):
        """
        Estimate the FLOPS based on the number of valid tokens in the current batch and the time taken.

        Args:
            batch_seqlens (List[int]): A list where each element represents the number of valid tokens in the current batch.
            delta_time (float): The time taken to process the batch, in seconds.

        Returns:
            estimated_flops (float): The estimated FLOPS based on the input tokens and time.
            promised_flops (float): The expected FLOPS of the current device.
        """
        tokens_sum = sum(batch_seqlens)
        func = self.estimate_func.get(self.config.model_type, self._estimate_unknown_flops)
        estimated_flops = func(tokens_sum, batch_seqlens, delta_time)
        promised_flops = get_device_flops()
        return estimated_flops, promised_flops
