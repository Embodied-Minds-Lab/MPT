from typing import Iterable, List, Optional, Tuple, cast

import torch
import torch.nn as nn
from torch import Tensor
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.siglip import SiglipVisionModel

import sglang.srt.managers.mm_utils as mm_utils
import sglang.srt.model_loader.weight_utils as weight_utils
import sglang.srt.utils as sgl_utils
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import MultiModalityDataPaddingPatternMultimodalTokens
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.llama import LlamaForCausalLM

# NaVILAConfig lives in a sglang-free module so it can be registered during sglang's early
# bootstrap without a circular import (see navila_config.py). The vision components are shared
# (sglang-free) so the HF training model can reuse them (see navila_vision.py).
from agent_system.models.navila_config import NaVILAConfig
from agent_system.models.navila_vision import DownSampleBlock, NaVILAProjector  # noqa: F401


class LlavaLlamaModel(nn.Module):
    """sglang inference class for NaVILA (VILA-llama3-8f). The class name equals
    config.architectures[0] so sglang's registry matches it."""

    def __init__(self, config, quant_config: Optional[QuantizationConfig] = None, prefix: str = "") -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.logits_processor = LogitsProcessor(config.text_config)
        self.llm = LlamaForCausalLM(config=config.text_config, quant_config=quant_config,
                                    prefix=sgl_utils.add_prefix("llm", prefix))
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.mm_projector = NaVILAProjector(config.mm_hidden_size, config.hidden_size,
                                            getattr(config, "torch_dtype", torch.float16))
        self.image_token_id = getattr(config, "image_token_index", -200)

    @property
    def dtype(self):
        return getattr(self.config, "torch_dtype", torch.float16)

    def forward(self, input_ids: Tensor, positions: Tensor, forward_batch: ForwardBatch,
                get_embedding: bool = False) -> LogitsProcessorOutput:
        output = mm_utils.general_mm_embed_routine(
            input_ids=input_ids, forward_batch=forward_batch, language_model=self.llm,
            data_embedding_funcs={Modality.IMAGE: self.get_image_feature},
            get_embedding=get_embedding, positions=positions,
        )
        return cast(LogitsProcessorOutput, output)

    def get_image_feature(self, mm_input: List[MultimodalDataItem]) -> Tensor:
        pixel_values = cast(Tensor, mm_input[0].feature)
        vto: BaseModelOutputWithPooling = self.vision_tower.__call__(
            pixel_values.to(device=self.vision_tower.device, dtype=self.vision_tower.dtype),
            output_hidden_states=True)
        sel = vto.hidden_states[self.config.mm_vision_select_layer]
        if self.config.mm_vision_select_feature != "cls_patch":
            raise NotImplementedError(self.config.mm_vision_select_feature)
        return self.mm_projector.__call__(sel)

    def load_weights(self, weights: Iterable[Tuple[str, Tensor]]) -> None:
        params = dict(self.named_parameters())
        for name, w in weights:
            if name.startswith("llm."):
                self.llm.load_weights([(name[len("llm."):], w)])
            elif name in params:
                p = params[name]
                getattr(p, "weight_loader", weight_utils.default_weight_loader)(p, w)
            elif name.startswith("vision_tower.") or name.startswith("mm_projector."):
                # some vision buffers are not in named_parameters; load straight into state_dict
                sd = self.state_dict()
                if name in sd:
                    sd[name].copy_(w)

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs) -> List[int]:
        return MultiModalityDataPaddingPatternMultimodalTokens().pad_input_tokens(input_ids, mm_inputs)


EntryClass = [LlavaLlamaModel]


def register():
    """Optional manual registration for standalone use. In a normal sglang run this is NOT
    needed: the AutoConfig side is done at sglang bootstrap (hf_transformers_utils.py patch,
    via navila_config) and the architecture is registered by sglang's auto-import of the
    srt/models/navila.py shim's EntryClass. Both steps here are idempotent."""
    from transformers import AutoConfig
    try:
        AutoConfig.register("llava_llama", NaVILAConfig)
    except Exception:
        pass  # already registered
    from sglang.srt.models.registry import ModelRegistry
    if "LlavaLlamaModel" not in ModelRegistry.models:
        ModelRegistry.models["LlavaLlamaModel"] = LlavaLlamaModel
