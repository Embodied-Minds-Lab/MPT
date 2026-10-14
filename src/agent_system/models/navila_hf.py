from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.siglip import SiglipVisionModel

from agent_system.models.navila_config import NaVILAConfig, NAVILA_IMAGE_TOKEN_ID
from agent_system.models.navila_vision import NaVILAProjector


class NavilaForCausalLM(PreTrainedModel):
    config_class = NaVILAConfig
    base_model_prefix = "llm"
    supports_gradient_checkpointing = True
    # FSDP auto-wrap needs a valid split list, or the whole 8B+vision model becomes one flat unit.
    _no_split_modules = ["LlamaDecoderLayer", "SiglipEncoderLayer"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(self, config: NaVILAConfig):
        super().__init__(config)
        self.llm = LlamaForCausalLM(config.text_config)
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.mm_projector = NaVILAProjector(config.mm_hidden_size, config.hidden_size)
        self.image_token_id = int(getattr(config, "train_image_token_id", NAVILA_IMAGE_TOKEN_ID))
        self.select_layer = int(getattr(config, "mm_vision_select_layer", -2))
        self.post_init()

    # ---- embeddings delegate to the llama backbone ----
    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.llm.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.llm.set_output_embeddings(new_embeddings)

    def gradient_checkpointing_enable(self, **kwargs):
        self.llm.gradient_checkpointing_enable(**kwargs)

    def _encode_images(self, pixel_values: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """pixel_values [num_frames, 3, H, W] -> vision embeddings [num_frames*tokens_per_frame, hidden].
        dtype is the target embedding dtype; device stays the input's (already on GPU under FSDP).
        Avoids self.vision_tower.device/.dtype which use next(parameters()) -> StopIteration under FSDP."""
        pv = pixel_values.to(dtype=dtype)
        feats = self.vision_tower(pv, output_hidden_states=True).hidden_states[self.select_layer]  # [F,729,1152]
        proj = self.mm_projector(feats)                       # [F,196,hidden]
        return proj.reshape(-1, proj.shape[-1])               # [F*196, hidden]

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            if pixel_values is not None:
                vis = self._encode_images(pixel_values, inputs_embeds.dtype).to(
                    device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                mask = input_ids == self.image_token_id                   # [B, seq]
                n_slots = int(mask.sum().item())
                if n_slots != vis.shape[0]:
                    raise ValueError(
                        f"NaVILA scatter mismatch: {n_slots} placeholder tokens (id "
                        f"{self.image_token_id}) in input_ids but {vis.shape[0]} vision rows "
                        f"(frames*tokens_per_frame). They must be equal.")
                inputs_embeds = inputs_embeds.clone()
                inputs_embeds[mask] = vis                                 # row-major, order-aligned
            input_ids = None  # embeds are authoritative from here
        return self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
