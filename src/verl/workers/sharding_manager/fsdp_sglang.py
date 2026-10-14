# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import logging
import os

import torch
import torch.distributed as dist
from sglang.srt.entrypoints.engine import Engine
try:
    from sglang.srt.model_executor.model_runner import LocalSerializedTensor
except ImportError:  # sglang main (0.5.20+): moved with the weight updater
    from sglang.srt.model_executor.model_runner_components.weight_updater import LocalSerializedTensor
from sglang.srt.utils import MultiprocessingSerializer
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

from verl import DataProto
from verl.protocol import all_gather_data_proto
from verl.utils.debug import GPUMemoryLogger, log_gpu_memory_usage
from verl.utils.fsdp_utils import fsdp_version, load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
from verl.utils.torch_functional import check_cuda_is_available

from .base import BaseShardingManager

# from vllm.distributed import parallel_state as sglang_ps
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# SGLANG_RELEASE_KV_ONLY=1: release only the KV cache between rollouts and keep the engine's weights resident.
# The default (release everything) unmaps the whole weights region; after resume, every tensor the model
# built at init that no checkpoint provides (gemma-4: the k_norm of KV-shared layers, derived buffers) holds
# garbage that update_weights never overwrites -> garbage rollouts (observed: per-tensor sync OK on a plain
# engine, BROKEN after release/resume). Costs the bf16 weights (~10 GB per engine) during
# training. Needs the tagged API (sglang >= 0.5.x); older engines ignore it.
# One helper for EVERY release/resume of the engine: the init-time release in sglang_rollout.py must use the
# same tags as the per-step release/resume here, or the first tagged resume leaves the weights unmapped and
# the first update_weights_from_tensor dies with cudaErrorIllegalAddress.
def sglang_memory_tags():
    if os.environ.get("SGLANG_RELEASE_KV_ONLY", "0") != "1":
        return None
    try:
        from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
        return [GPU_MEMORY_TYPE_KV_CACHE]
    except ImportError:
        return None


def _preprocess_tensor_for_update_weights(tensor: torch.Tensor):
    if isinstance(tensor, DTensor):
        return tensor.full_tensor()
    return tensor


class FSDPSGLangShardingManager(BaseShardingManager):
    @check_cuda_is_available()
    def __init__(
        self,
        module: FSDP,
        inference_engine: Engine,
        model_config,
        full_params: bool = False,
        device_mesh: DeviceMesh = None,
        offload_param: bool = False,
    ):
        self.module = module
        self.inference_engine = inference_engine
        self.model_config = model_config
        self.device_mesh = device_mesh
        self.offload_param = offload_param

        # Full params
        self.full_params = full_params
        if full_params and fsdp_version(self.module) == 1:
            FSDP.set_state_dict_type(self.module, state_dict_type=StateDictType.FULL_STATE_DICT, state_dict_config=FullStateDictConfig())
        elif fsdp_version(self.module) == 1:
            FSDP.set_state_dict_type(
                self.module,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        self.tp_size = self.device_mesh["infer_tp"].size()
        self.tp_rank = self.device_mesh["infer_tp"].get_local_rank()

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = torch.cuda.get_rng_state()
        # get a random rng states
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh["dp"].get_local_rank()
            torch.cuda.manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

    @GPUMemoryLogger(role="FSDPSGLangShardingManager enter", logger=logger)
    def __enter__(self):
        torch.cuda.empty_cache()
        log_gpu_memory_usage("Before state_dict() in sharding manager memory", logger=logger)
        if self.offload_param:
            load_fsdp_model_to_gpu(self.module)
        params = self.module.state_dict()
        log_gpu_memory_usage("After state_dict() in sharding manager memory", logger=logger)
        device = torch.cuda.current_device()  # used when fsdp2 set cpu_offload_policy
        params = {k: v.to(device, non_blocking=True) if fsdp_version(self.module) == 2 else v for k, v in params.items()}
        # Copy, not share memory
        self.update_weights(params)
        log_gpu_memory_usage("After sync model weights in sharding manager", logger=logger)
        self._post_sync_check(params)

        del params
        if self.offload_param:
            offload_fsdp_model_to_cpu(self.module)
        torch.cuda.empty_cache()
        log_gpu_memory_usage("After del state_dict and empty_cache in sharding manager", logger=logger)

        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.gen_random_states)

    @GPUMemoryLogger(role="FSDPSGLangShardingManager exit", logger=logger)
    def __exit__(self, exc_type, exc_value, traceback):
        log_gpu_memory_usage("Before SGLang offload in sharding manager", logger=logger)
        self.release_memory()
        log_gpu_memory_usage("After SGLang offload in sharding manager", logger=logger)

        self.module.train()

        # add empty cache after each compute
        torch.cuda.empty_cache()

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)

    # architectures whose sglang params_dict keeps the legacy (pre-4.49) checkpoint naming
    _LEGACY_VL_NAMING = ("qwen2_vl", "qwen2_5_vl", "qwen3_vl", "qwen3_vl_moe")

    def _hf_to_sglang_name(self, name: str) -> str:
        """transformers >= ~4.49 restructured Qwen2-VL/Qwen2.5-VL as model.visual.* +
        model.language_model.*, while sglang's params_dict keeps the legacy checkpoint naming
        (visual.*, model.*). Remap at the runtime weight-sync boundary; text models pass through.
        Only for the Qwen-VL family: sglang's gemma4_mm keeps HF's model.language_model.* /
        model.vision_tower.* layout (its load_weights strips the leading `model.` itself), so the
        Qwen rule would silently drop every gemma-4 tensor from the sync."""
        if getattr(self.model_config, "model_type", None) not in self._LEGACY_VL_NAMING:
            return name
        if name.startswith("model.visual."):
            return name.replace("model.visual.", "visual.", 1)
        if name.startswith("model.language_model."):
            return name.replace("model.language_model.", "model.", 1)
        return name

    def _post_sync_check(self, params):
        """VERL_SYNC_CHECK=1: after the sync, greedily generate a few tokens on a fixed prompt and log the text
        together with the pushed tensor count and the checksum of the embedding table -- the engine started on
        dummy weights, so garbage here means the sync did not land (the rollouts would be garbage too)."""
        if os.environ.get("VERL_SYNC_CHECK", "0") != "1" or self.device_mesh["infer_tp"].get_local_rank() != 0:
            return
        try:
            emb = [v for k, v in params.items() if k.endswith("embed_tokens.weight")]
            csum = float(_preprocess_tensor_for_update_weights(emb[0]).float().sum().item()) if emb else float("nan")
            out = self.inference_engine.generate(
                prompt="<bos><start_of_turn>user\nWhat is the capital of France? Answer in one word.<end_of_turn>\n<start_of_turn>model\n",
                sampling_params={"temperature": 0, "max_new_tokens": 8},
            )
            text = out["text"] if isinstance(out, dict) else out
            print(f"[fsdp_sglang] post-sync check: pushed {len(params)} tensors, embed_tokens sum={csum:.4f}, greedy -> {text!r}", flush=True)
        except Exception as e:  # diagnostics must never break the run
            print(f"[fsdp_sglang] post-sync check failed: {type(e).__name__}: {e}", flush=True)

    def _memory_tags(self):
        return sglang_memory_tags()

    def _resume_memory(self):
        tags = self._memory_tags()
        if tags is None:
            return self.inference_engine.resume_memory_occupation()
        return self.inference_engine.resume_memory_occupation(tags=tags)

    def update_weights(self, params):
        if self.device_mesh["infer_tp"].get_local_rank() == 0:
            self._resume_memory()

        # Most naive implementation, can optimize a lot if it is bottleneck from sglang Engine weight update
        named_tensors = [(self._hf_to_sglang_name(k), v) for k, v in params.items()]
        load_format = None
        for tensor_index, (name, tensor) in enumerate(named_tensors):
            serialized_tensor = MultiprocessingSerializer.serialize(_preprocess_tensor_for_update_weights(tensor))

            if self.device_mesh["infer_tp"].get_local_rank() == 0:
                gathered_serialized_tensors = [None for _ in range(self.device_mesh["infer_tp"].mesh.size()[0])]
            else:
                gathered_serialized_tensors = None
            dist.gather_object(
                obj=serialized_tensor,
                object_gather_list=gathered_serialized_tensors,
                dst=self.device_mesh["infer_tp"].mesh.tolist()[0],
                group=self.device_mesh["infer_tp"].get_group(),
            )

            if self.device_mesh["infer_tp"].get_local_rank() == 0:
                ret = self.inference_engine.update_weights_from_tensor(
                    named_tensors=[
                        (
                            name,
                            LocalSerializedTensor(values=gathered_serialized_tensors),
                        )
                    ],
                    load_format=load_format,
                    flush_cache=tensor_index == len(named_tensors) - 1,
                )
                # sglang reports (success, message) and the engine keeps whatever it had (random weights
                # under load_format=dummy) on failure -- silently, so a refused sync means garbage rollouts.
                # Never let that pass: fail loudly on the first refused tensor.
                if isinstance(ret, (tuple, list)) and len(ret) >= 1 and ret[0] is False:
                    raise RuntimeError(f"[fsdp_sglang] update_weights_from_tensor refused '{name}': {ret[1] if len(ret) > 1 else ret}")
                if isinstance(ret, dict) and ret.get("success") is False:
                    raise RuntimeError(f"[fsdp_sglang] update_weights_from_tensor refused '{name}': {ret.get('message', ret)}")

    def release_memory(self):
        if self.device_mesh["infer_tp"].get_local_rank() == 0:
            tags = self._memory_tags()
            if tags is None:
                self.inference_engine.release_memory_occupation()
            else:
                self.inference_engine.release_memory_occupation(tags=tags)

    def preprocess_data(self, data: DataProto) -> DataProto:
        """All gather across tp group to make each rank has identical input."""
        if self.tp_size == 1:
            return data

        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        group = self.device_mesh["infer_tp"].get_group()

        all_gather_data_proto(data=data, process_group=group)
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        """Get chunk data of this tp rank since we do all gather in preprocess."""
        if self.tp_size == 1:
            return data

        return data.chunk(chunks=self.tp_size)[self.tp_rank]
