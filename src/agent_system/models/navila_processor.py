import math
from typing import Any, Dict, List, Optional, Type

import torch
import torch.nn as nn
from PIL import Image

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)

from agent_system.models.navila_sglang import LlavaLlamaModel
# Shared placeholder-token constants (sglang-free module) so serving + training agree.
from agent_system.models.navila_config import (
    NAVILA_IMAGE_TOKEN, NAVILA_IMAGE_TOKEN_ID, NAVILA_TOKENS_PER_FRAME)


def _tokens_per_frame(hf_config) -> int:
    """Number of LLM tokens one frame occupies after the 2x2 mlp_downsample projector.
    siglip 384/patch14 -> 27x27=729 patches -> pad odd side to 28 -> 14x14 = 196."""
    vc = hf_config.vision_config
    side = vc.image_size // vc.patch_size            # 27
    padded = side + (side % 2)                        # 28
    return (padded // 2) * (padded // 2)              # 196


def _flatten_frames(image_data) -> List[Image.Image]:
    """image_data for one request is a list of PIL frames (possibly nested one level)."""
    frames: List[Image.Image] = []
    if image_data is None:
        return frames
    stack = list(image_data) if isinstance(image_data, (list, tuple)) else [image_data]
    for it in stack:
        if isinstance(it, (list, tuple)):
            frames.extend(_flatten_frames(it))
        else:
            frames.append(it)
    return frames


class NaVILAMultimodalProcessor(BaseMultimodalProcessor):
    """Registered against the LlavaLlamaModel architecture so sglang's get_mm_processor
    resolves it for hf_config.architectures == ["LlavaLlamaModel"]."""

    models: List[Type[nn.Module]] = [LlavaLlamaModel]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs) -> None:
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        # _processor is the HF NaVILAProcessor (tokenizer + siglip image_processor). Be tolerant
        # of it being a bare tokenizer, mirroring internvl.py.
        if hasattr(_processor, "image_processor") and _processor.image_processor is not None:
            self.image_processor = _processor.image_processor
        else:
            from transformers import SiglipImageProcessor
            self.image_processor = SiglipImageProcessor(
                do_normalize=True, do_rescale=True, do_resize=True,
                image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5],
                rescale_factor=1 / 255, resample=3,
                size={"height": hf_config.vision_config.image_size,
                      "width": hf_config.vision_config.image_size})
        self.tokenizer = getattr(_processor, "tokenizer", _processor)
        self.tokens_per_frame = _tokens_per_frame(hf_config)
        self.image_token_id = NAVILA_IMAGE_TOKEN_ID
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=NAVILA_IMAGE_TOKEN,
            image_token_id=NAVILA_IMAGE_TOKEN_ID,
        ).build(_processor)

    def _pixel_values(self, frames: List[Image.Image]) -> torch.Tensor:
        rgb = [f.convert("RGB") if isinstance(f, Image.Image) else f for f in frames]
        out = self.image_processor(images=rgb, return_tensors="pt")
        return out["pixel_values"]  # [num_frames, 3, H, W]

    def _expand_ids(self, input_text, num_frames: int) -> List[int]:
        """Expand each single image-placeholder token into tokens_per_frame copies.
        Requires exactly num_frames placeholders (one per <image>), one 196-run per frame."""
        n = self.tokens_per_frame
        if isinstance(input_text, str):
            ids = self.tokenizer(input_text, add_special_tokens=False)["input_ids"]
        else:
            ids = list(input_text)
        placeholders = [i for i, t in enumerate(ids) if t == self.image_token_id]
        if len(placeholders) != num_frames:
            raise ValueError(
                f"NaVILA processor: found {len(placeholders)} image placeholders "
                f"(id {self.image_token_id}) but {num_frames} frames; must be equal.")
        out: List[int] = []
        for i, t in enumerate(ids):
            out.extend([self.image_token_id] * n if t == self.image_token_id else [t])
        return out

    async def process_mm_data_async(self, image_data, input_text, request_obj, **kwargs
                                    ) -> Optional[Dict[str, Any]]:
        frames = _flatten_frames(image_data)
        if not frames:
            return None
        pixel_values = self._pixel_values(frames)
        ids = self._expand_ids(input_text, num_frames=len(frames))
        input_ids_tensor = torch.tensor(ids, dtype=torch.long)
        offsets = self.get_mm_items_offset(
            input_ids=input_ids_tensor, mm_token_id=self.image_token_id)
        items = [MultimodalDataItem(
            feature=pixel_values, modality=Modality.IMAGE, offsets=offsets)]
        return {
            "input_ids": ids,
            "mm_items": items,
            "im_token_id": self.image_token_id,
        }
