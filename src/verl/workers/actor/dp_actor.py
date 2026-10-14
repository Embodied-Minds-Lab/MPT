# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
import itertools
import logging
import os
from typing import Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_answer_preference_loss, compute_policy_loss, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs, ulysses_pad
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from verl.utils.flash_attn_compat import index_first_axis, pad_input, rearrange, unpad_input  # flash_attn, or transformers/einops on an FA4-only env
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        # mixed-modality chain society: text agents' rows are tiled to (bs, 3, seq) purely so the
        # CROSS-AGENT collate stays rectangular with VLM agents' mrope rows. Whether 3D position
        # ids are REAL mrope depends on the model this actor serves, so decide once here.
        try:
            _mcfg = getattr(actor_module, "config", None) or getattr(getattr(actor_module, "module", None), "config", None)
            self._is_mrope = getattr(_mcfg, "model_type", "") in ("qwen2_vl", "qwen2_5_vl")
        except Exception:
            self._is_mrope = False
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )
        self.device_name = get_device_name()

    def _accepts_logits_to_keep(self) -> bool:
        """Cached: does the wrapped HF model's forward take `logits_to_keep`? (FSDP wraps the module; look through it.)"""
        cached = getattr(self, "_logits_to_keep_ok", None)
        if cached is None:
            import inspect
            mod = self.actor_module
            mod = getattr(mod, "_fsdp_wrapped_module", None) or getattr(mod, "module", None) or mod
            try:
                cached = "logits_to_keep" in inspect.signature(mod.forward).parameters
            except (TypeError, ValueError):
                cached = False
            self._logits_to_keep_ok = cached
            print(f"[dp_actor] logits_to_keep {'supported' if cached else 'NOT supported'} by {type(mod).__name__}", flush=True)
        return cached

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False, extra_out=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)

        extra_out: optional dict of flags for the answer-level preference loss. With None the
            forward is the plain one. {"no_inplace_backward": True} disables the in-place
            backward of the log-prob computation (log_prob then feeds two loss terms).
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:
                if self._is_mrope:  # qwen2vl mrope
                    position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)
                else:
                    # text model in a mixed society: the 3 channels are identical tiles — undo
                    position_ids = position_ids[:, 0]

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    # The answer-level preference loss (bid_pref_scope="answer") feeds
                    # log_prob into BOTH the PG loss and the preference loss. Defensively
                    # disable the in-place backward (see the note in the non-rmpad branch
                    # below); without the marker the forward is the plain one.
                    if extra_out is not None and extra_out.get("no_inplace_backward", False):
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                # PATCH: only the response positions are ever read below (logits[:, -R-1:-1]), so ask
                # the model for just those. Without this the padded path materialises (bsz, prompt+response, vocab)
                # logits -- gemma-4: 20480 x 262k x fp32 = 21.5 GB per sequence; two sequences in a micro-batch
                # asked for 40 GiB and OOMed compute_log_prob (observed in a deep-research smoke run). HF models expose
                # `logits_to_keep=N` (last N positions); models without it keep the old full-width path.
                if not self.use_fused_kernels and self._accepts_logits_to_keep():
                    extra_args["logits_to_keep"] = response_length + 1
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    # The in-place backward stays off when log_prob feeds two loss terms (the
                    # answer-level preference loss marks this with no_inplace_backward):
                    # flash-attn's cross_entropy_loss writes its gradient back into the logits
                    # buffer, and two gradient branches on one buffer would silently corrupt
                    # each other. Without the marker the forward is the plain one.
                    _no_ib = extra_out is not None and extra_out.get("no_inplace_backward", False)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"], inplace_backward=not _no_ib)
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len, count_padding=not self.use_remove_padding)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy)
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        wg_id = data.non_tensor_batch['wg_id'][0]

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Answer-level preference path (bid_pref_scope="answer"): the pair keys must be
        # selected into select_keys explicitly, because data.select(batch_keys=...) below is
        # whitelist-semantics -- unlisted keys are stripped, the pair keys would then never
        # exist in the mini/micro batches, and the preference loss would silently degrade to
        # plain GRPO with zero errors. With the switch on but no pair keys injected by the
        # collector, raise immediately (the no-silent-failure rule).
        # The pair encoding is content-addressed: pref_pair_id (int64, a pair's two rows
        # share one id, non-member rows carry -1) + pref_pair_role (+1 winner / -1 loser /
        # 0 non-member). Pairs are matched by id inside each micro-batch (see the loss site
        # below), so balance_batch's global reorder, DP-rank chunking, mini/micro splits and
        # dynamic-bsz permutations can never point a pair at the wrong rows -- at worst a
        # pair lands in two micro-batches, and such pairs are counted, warned about and
        # skipped, never mismatched.
        if self.config.get("bid_pref_enable", False):
            if self.config.get("bid_pref_scope", "answer") != "answer":
                raise ValueError(f"bid_pref_scope accepts only 'answer' in this release, "
                                 f"got {self.config.get('bid_pref_scope')!r}")
            _pair_missing = [k for k in ("pref_pair_id", "pref_pair_role") if k not in data.batch.keys()]
            if _pair_missing:
                raise KeyError(
                    f"bid_pref_enable=True with bid_pref_scope='answer', but DataProto.batch lacks "
                    f"the pair keys {_pair_missing} (did the collector inject the counterfactual "
                    f"pair rows?). Refusing to silently skip the answer-level preference loss."
                )
            select_keys.extend(["pref_pair_id", "pref_pair_role"])
            if "pref_pair_weight" in data.batch.keys():
                select_keys.append("pref_pair_weight")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if self.config.use_adaptive_ppo_mini_batch_size:
            self.config.ppo_mini_batch_size = data.meta_info.get(f"{wg_id}/ppo_mini_batch_size", self.config.ppo_mini_batch_size)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len, count_padding=not self.use_remove_padding)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                # Counterfactual preference iterations (bid_pref_scope="answer" under the
                # act/preference alternation) build the WHOLE mini-batch out of pair rows,
                # placed adjacently by the collector (balance_batch is skipped for such
                # batches in ray_trainer). The MC memory setting ppo_micro_batch_size_per_gpu=1
                # would cut every pair in half and the answer loss would warn-and-skip 100% of
                # pairs, so pure-preference mini-batches are re-chunked at the smallest EVEN
                # micro size >= the configured one. Mixed batches are left alone -- the
                # alternation design never produces them, and a stray one degrades to the
                # loud warn+skip path rather than to silence.
                if (self.config.get("bid_pref_enable", False)
                        and self.config.get("bid_pref_scope", "answer") == "answer"
                        and "pref_pair_role" in mini_batch.batch.keys()
                        and mini_batch.batch.batch_size[0] > 0
                        and bool((mini_batch.batch["pref_pair_role"] != 0).all())):
                    _even = max(2, ((self.config.ppo_micro_batch_size_per_gpu + 1) // 2) * 2)
                    _bs = mini_batch.batch.batch_size[0]
                    if _bs % _even == 0:
                        self.gradient_accumulation = _bs // _even
                        if has_multi_modal_inputs:
                            micro_batches = mini_batch.select(select_keys, non_tensor_select_keys).chunk(self.gradient_accumulation)
                        else:
                            micro_batches = mini_batch.split(_even)
                        if not getattr(self, "_pairmicro_logged", False):
                            print(f"[AnswerPref] pure-preference mini-batch: re-chunked to "
                                  f"micro={_even} ({self.gradient_accumulation} micro-batches) "
                                  f"so pairs stay co-resident", flush=True)
                            self._pairmicro_logged = True
                    else:
                        print(f"[AnswerPref] WARN: pure-preference mini-batch of {_bs} rows is "
                              f"not divisible by {_even} -- keeping the configured micro split; "
                              f"straddling pairs will be counted and skipped", flush=True)

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_torch_device().current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(get_torch_device().current_device())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # Counterfactual answer-level preference loss (the bid phase of MPT). With
                    # bid_pref_enable=False none of the branches below are taken, extra_out stays
                    # None and the forward is the plain PG path.
                    bid_pref_enable = self.config.get("bid_pref_enable", False)
                    bid_pref_coef = self.config.get("bid_pref_coef", 1.0)
                    bid_pref_beta = self.config.get("bid_pref_beta", 1.0)
                    # bid_pref_ref_anchor (answer scope only): anchor the sequence margin on the
                    # sampling-time log-probs of the same rows (data["old_log_probs"]) -- the DPO
                    # reference term with the sampling policy as reference, so the first update
                    # starts at margin 0 instead of at the answers' unconditional length /
                    # fluency difference (see core_algos.compute_answer_preference_loss). Off by
                    # default: the unanchored path is numerically identical to before.
                    bid_pref_ref_anchor = bool(self.config.get("bid_pref_ref_anchor", False))
                    # bid_pref_scope: "answer" = answer-level sequence Bradley-Terry
                    # (core_algos.compute_answer_preference_loss); pair rows are injected by the
                    # counterfactual collector, pairing runs over the content-addressed batch
                    # keys "pref_pair_id"/"pref_pair_role" (encoding notes at the select_keys
                    # site). It is the only scope in this release.
                    bid_pref_scope = self.config.get("bid_pref_scope", "answer")
                    if bid_pref_enable and bid_pref_scope != "answer":
                        raise ValueError(f"bid_pref_scope accepts only 'answer', got {bid_pref_scope!r}")
                    extra_out = None
                    if bid_pref_enable and bid_pref_scope == "answer" and "pref_pair_id" in data.keys():
                        # The answer-level path consumes only the forward's (bs, resp_len)
                        # log_prob and needs no alt_ids alignment, so both rmpad and fused
                        # layouts are compatible. The marker makes the forward defensively
                        # disable the in-place backward (log_prob feeds both the PG and the
                        # preference loss; see the _forward_micro_batch note).
                        extra_out = {"no_inplace_backward": True}
                        if not getattr(self, "_answerpref_logged", False):
                            print(f"[AnswerPref] answer-level preference loss ACTIVE: coef={bid_pref_coef}, beta={bid_pref_beta}, "
                                  f"ref_anchor={bid_pref_ref_anchor} (old_log_probs as sampling-policy reference)", flush=True)
                            self._answerpref_logged = True

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy, extra_out=extra_out)

                    # Answer path: pair member rows (pref_pair_role != 0) are the
                    # collector-injected counterfactual rows and serve ONLY the preference
                    # loss (advantage already zeroed). They must leave PG/entropy/KL as WHOLE
                    # ROWS -- both the loss terms and the denominators: otherwise
                    # entropy_coeff != 0 would entropy-maximize the y_l text, use_kl_loss
                    # would tax it with KL, and token-mean aggregation would dilute the PG
                    # loss. Removal slices whole rows rather than zeroing loss_mask because
                    # the seq-mean-* aggregations divide by mask.sum(dim=-1) per row and an
                    # all-zero row divides into NaN. _pg_sel is the identity <=> the feature
                    # is off or this micro-batch has no member rows, in which case every loss
                    # below is bit-identical to before this change.
                    _pg_keep = None
                    if bid_pref_enable and bid_pref_scope == "answer" and "pref_pair_role" in data.keys():
                        _cf_rows = data["pref_pair_role"].long() != 0
                        if bool(_cf_rows.any()):
                            _pg_keep = torch.nonzero(~_cf_rows).flatten()
                            append_to_dict(metrics, {f"actor/{wg_id}/answer_pref_cf_rows": float(int(_cf_rows.sum().item()))})
                    _pg_sel = (lambda t: t) if _pg_keep is None else (lambda t: t[_pg_keep])
                    _pg_empty = _pg_keep is not None and _pg_keep.numel() == 0

                    if _pg_empty:
                        # the whole micro-batch is injected rows: PG is 0 (graph-connected so
                        # DDP does not blow up) and only the preference term remains
                        _z = log_prob.sum() * 0.0
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = _z, _z.detach(), _z.detach(), _z.detach()
                    else:
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                            old_log_prob=_pg_sel(old_log_prob),
                            log_prob=_pg_sel(log_prob),
                            advantages=_pg_sel(advantages),
                            response_mask=_pg_sel(response_mask),
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                            loss_agg_mode=loss_agg_mode,
                        )

                    if entropy_coeff != 0:
                        if _pg_empty:
                            entropy_loss = log_prob.sum() * 0.0
                        else:
                            entropy_loss = agg_loss(loss_mat=_pg_sel(entropy), loss_mask=_pg_sel(response_mask), loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss (injected rows leave KL's loss term and denominator too)
                        if _pg_empty:
                            kl_loss = log_prob.sum() * 0.0
                        else:
                            kld = kl_penalty(logprob=_pg_sel(log_prob), ref_logprob=_pg_sel(ref_log_prob), kl_penalty=self.config.kl_loss_type)
                            kl_loss = agg_loss(loss_mat=kld, loss_mask=_pg_sel(response_mask), loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics[f"actor/{wg_id}/kl_loss"] = kl_loss.detach().item()
                        metrics[f"actor/{wg_id}/kl_coef"] = self.config.kl_loss_coef

                    if bid_pref_enable and bid_pref_scope == "answer" and "pref_pair_id" in data.keys():
                        # Pair recovery (content-addressed): a pair's two rows share one
                        # pref_pair_id; pref_pair_role=+1 marks the winner row, -1 the loser,
                        # 0 non-members. Matching happens by id WITHIN this micro-batch with
                        # no positional assumptions, so balance_batch reorders, mini/micro
                        # splits and rearrange_micro_batches permutations can never point a
                        # pair at the wrong rows. The collector's contract (see the
                        # counterfactual_driver module docstring): a pair's two rows are
                        # placed adjacently and land in the same micro-batch chunk; pairs
                        # still split apart are counted, warned about and skipped here, never
                        # force-matched. Both rows are the same agent's full candidate answers
                        # for the same context (advantage zeroed, loss_mask normal, serving
                        # only this loss; PG/entropy/KL removal happened above).
                        _pid = data["pref_pair_id"].long()
                        _role = data["pref_pair_role"].long()
                        _w_rows = torch.nonzero((_pid >= 0) & (_role == 1)).flatten().tolist()
                        _l_rows = torch.nonzero((_pid >= 0) & (_role == -1)).flatten().tolist()
                        _w_map = {int(_pid[i]): i for i in _w_rows}
                        _l_map = {int(_pid[i]): i for i in _l_rows}
                        if len(_w_map) != len(_w_rows) or len(_l_map) != len(_l_rows):
                            raise ValueError(
                                f"pref_pair_id repeats within one micro-batch for the same role "
                                f"(winner rows {len(_w_rows)}/{len(_w_map)} ids, loser rows {len(_l_rows)}/{len(_l_map)} ids) "
                                "-- the collector's injected encoding is broken; refusing to "
                                "guess the pairing, never silent."
                            )
                        _matched = sorted(set(_w_map) & set(_l_map))
                        _n_split = len(set(_w_map) ^ set(_l_map))
                        if _n_split:
                            print(f"[AnswerPref] WARN: {_n_split} pair(s) have their two rows split "
                                  f"across micro-batches; skipping them here (the collector should "
                                  f"emplace pair rows adjacently before chunking)", flush=True)
                            append_to_dict(metrics, {f"actor/{wg_id}/answer_pref_pairs_split": float(_n_split)})
                        if _matched:
                            w_idx = torch.tensor([_w_map[i] for i in _matched], dtype=torch.long, device=log_prob.device)
                            l_idx = torch.tensor([_l_map[i] for i in _matched], dtype=torch.long, device=log_prob.device)
                            # per-pair weight read from the winner row (injection contract:
                            # both rows of a pair carry the same value)
                            ap_weight = data["pref_pair_weight"][w_idx] if "pref_pair_weight" in data.keys() else None
                            # Reference anchor: the sampling-time log-probs of the SAME
                            # micro-batch rows. Key "old_log_probs" is selected unconditionally
                            # into select_keys and read as old_log_prob above for the PPO
                            # ratio; ray_trainer recomputes it over the full sub-batch AFTER
                            # the collector injected the pair rows, so pure-preference
                            # mini-batches carry it too. The absent-key branch is defensive:
                            # fall back to the unanchored margin and say so once.
                            _ap_old = None
                            if bid_pref_ref_anchor:
                                if "old_log_probs" in data.keys():
                                    _ap_old = old_log_prob
                                elif not getattr(self, "_anchor_missing_logged", False):
                                    print("[AnswerPref] WARN: bid_pref_ref_anchor=True but 'old_log_probs' is "
                                          "absent from this micro-batch -- falling back to the unanchored "
                                          "margin (logged once)", flush=True)
                                    self._anchor_missing_logged = True
                            ap_loss, ap_n, ap_sat, ap_info = compute_answer_preference_loss(
                                log_prob=log_prob,
                                w_idx=w_idx,
                                l_idx=l_idx,
                                response_mask_w=response_mask[w_idx],
                                response_mask_l=response_mask[l_idx],
                                pair_weight=ap_weight,
                                beta=bid_pref_beta,
                                old_log_prob=_ap_old,
                                ref_anchor=bid_pref_ref_anchor and _ap_old is not None,
                                return_margins=True,
                            )
                            policy_loss = policy_loss + ap_loss * bid_pref_coef
                            # answer_pref_bigmargin_frac: share of pairs with |m| above the
                            # reporting threshold (the margin is no longer clamped, see
                            # core_algos.compute_answer_preference_loss; monitoring only).
                            # answer_pref_margin_raw_mean: mean over valid pairs of the raw
                            # margin lp_w - lp_l; answer_pref_margin_mean: mean of the margin
                            # that entered the loss (anchored when ref_anchor took effect,
                            # else identical to the raw one).
                            _ap_valid = ap_info["valid"]
                            _ap_mraw = ap_info["margins_raw"]
                            _ap_manc = ap_info["margins"]
                            _ap_has_valid = bool(_ap_valid.numel() > 0 and _ap_valid.any())
                            append_to_dict(metrics, {f"actor/{wg_id}/answer_pref_loss": ap_loss.detach().item(),
                                                     f"actor/{wg_id}/answer_pref_pairs": float(ap_n),
                                                     f"actor/{wg_id}/answer_pref_bigmargin_frac": ap_sat,
                                                     f"actor/{wg_id}/answer_pref_margin_raw_mean":
                                                         float(_ap_mraw[_ap_valid].mean().item()) if _ap_has_valid else 0.0,
                                                     f"actor/{wg_id}/answer_pref_margin_mean":
                                                         float(_ap_manc[_ap_valid].mean().item()) if _ap_has_valid else 0.0,
                                                     f"actor/{wg_id}/answer_pref_ref_anchor": float(bool(ap_info["anchored"]))})
                            # Per-pair inspection dump (MPT_CF_MARGIN_DUMP=<path>.jsonl; the
                            # older MC_CF_MARGIN_DUMP is accepted too): the
                            # pre-update sequence log-prob margin of every matched pair. A
                            # margin > 0 means the policy already prefers the branch-winning
                            # answer; watching margins on FRESH pairs across preference
                            # iterations is the direct "is it learning the right direction"
                            # signal. "margins_raw" is lp_w - lp_l; "margins" is the margin
                            # that entered the loss (anchored on logp_*_old when ref_anchor
                            # is on, else equal to margins_raw). "pid" tells relaunched
                            # trainer processes apart. Observational only; write errors are
                            # swallowed.
                            import os as _os
                            _mdump = (_os.environ.get("MPT_CF_MARGIN_DUMP")
                                      or _os.environ.get("MC_CF_MARGIN_DUMP", ""))
                            if _mdump:
                                try:
                                    import json as _json
                                    import time as _time
                                    _lw = ap_info["lp_w"]
                                    _ll = ap_info["lp_l"]
                                    _lw_old = ap_info["lp_w_old"]
                                    _ll_old = ap_info["lp_l_old"]
                                    _rec = {"ts": _time.time(), "pid": _os.getpid(), "wg_id": str(wg_id),
                                            "pair_ids": [int(x) for x in _matched],
                                            "logp_w": [round(float(x), 4) for x in _lw],
                                            "logp_l": [round(float(x), 4) for x in _ll],
                                            "logp_w_old": ([round(float(x), 4) for x in _lw_old]
                                                           if _lw_old is not None else None),
                                            "logp_l_old": ([round(float(x), 4) for x in _ll_old]
                                                           if _ll_old is not None else None),
                                            "margins_raw": [round(float(x), 4) for x in _ap_mraw],
                                            "margins": [round(float(x), 4) for x in _ap_manc],
                                            "ref_anchor": bool(ap_info["anchored"]),
                                            "weights": ([round(float(x), 4) for x in ap_weight]
                                                        if ap_weight is not None else None),
                                            "loss": round(float(ap_loss.detach().item()), 6),
                                            "beta": float(bid_pref_beta)}
                                    with open(_mdump, "a") as _f:
                                        _f.write(_json.dumps(_rec) + "\n")
                                except Exception as _e:  # noqa: BLE001
                                    print(f"[AnswerPref] margin dump failed: {_e}", flush=True)

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    data = {
                        f"actor/{wg_id}/pg_loss": pg_loss.detach().item(),
                        f"actor/{wg_id}/pg_clipfrac": pg_clipfrac.detach().item(),
                        f"actor/{wg_id}/ppo_kl": ppo_kl.detach().item(),
                        f"actor/{wg_id}/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    }
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {f"actor/{wg_id}/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
