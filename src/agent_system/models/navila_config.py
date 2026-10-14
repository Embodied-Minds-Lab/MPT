from transformers.configuration_utils import PretrainedConfig
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.siglip import SiglipVisionConfig
from transformers.processing_utils import ProcessorMixin

# The image placeholder used across serving + training: a real llama3 reserved special token
# (in-vocab, positive id) that both the sglang processor and the HF training model scatter vision
# embeddings at. NOT the HF -200 sentinel (which cannot be embedded / positively indexed).
NAVILA_IMAGE_TOKEN = "<|reserved_special_token_0|>"
NAVILA_IMAGE_TOKEN_ID = 128002
# LLM tokens one frame occupies after the 2x2 mlp_downsample projector (siglip 384/patch14: 729 -> 196).
NAVILA_TOKENS_PER_FRAME = 196

_NAVILA_IMG_PROC = None


def navila_image_processor():
    """The siglip image processor NaVILA uses (384x384, mean/std 0.5, bicubic, rescale 1/255),
    cached. Built directly so the client-side preprocess_fn does not depend on an AutoProcessor
    being registered/loaded in its process. Identical config to the sglang NaVILA processor, so
    training and serving produce the same pixel_values."""
    global _NAVILA_IMG_PROC
    if _NAVILA_IMG_PROC is None:
        from transformers import SiglipImageProcessor
        _NAVILA_IMG_PROC = SiglipImageProcessor(
            do_normalize=True, do_rescale=True, do_resize=True,
            image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5],
            rescale_factor=1 / 255, resample=3, size={"height": 384, "width": 384})
    return _NAVILA_IMG_PROC


class NaVILAConfig(PretrainedConfig):
    """Registered with transformers AutoConfig under model_type=llava_llama so sglang's
    get_config() does not choke on the unknown type. The text/vision sub-configs are
    materialized as objects."""

    model_type = "llava_llama"
    sub_configs = {"text_config": LlamaConfig, "vision_config": SiglipVisionConfig}
    _auto_class = "AutoConfig"

    def __init__(self, text_config=None, vision_config=None, *, hidden_size=4096,
                 mm_hidden_size=1152, mm_projector_type="mlp_downsample",
                 mm_vision_select_feature="cls_patch", mm_vision_select_layer=-2,
                 num_video_frames=8, image_token_index=-200, **kwargs):
        super().__init__(**kwargs)
        self.text_config = LlamaConfig(**text_config) if isinstance(text_config, dict) else (text_config or LlamaConfig())
        self.vision_config = SiglipVisionConfig(**vision_config) if isinstance(vision_config, dict) else (vision_config or SiglipVisionConfig())
        self.hidden_size = hidden_size
        self.mm_hidden_size = mm_hidden_size
        self.mm_projector_type = mm_projector_type
        self.mm_vision_select_feature = mm_vision_select_feature
        self.mm_vision_select_layer = mm_vision_select_layer
        self.num_video_frames = num_video_frames
        self.image_token_index = image_token_index


class NaVILAProcessor(ProcessorMixin):
    """Minimal HF processor bundling the siglip image_processor + the llama3 tokenizer, so that
    sglang's get_processor() (AutoProcessor.from_pretrained on the checkpoint dir) succeeds --
    it raises for a tokenizer-only dir. The real image-token expansion happens in the sglang
    NaVILAMultimodalProcessor; this class just carries `.tokenizer` and `.image_processor`."""

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "SiglipImageProcessor"
    tokenizer_class = "PreTrainedTokenizerFast"

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        super().__init__(image_processor, tokenizer)


def register_navila_hf(with_model: bool = False):
    """Register NaVILA's config + processor (+ optionally the training model) with transformers
    Auto* factories. sglang-free and idempotent, so it can run at sglang's early bootstrap in every
    (spawned) process. Pass with_model=True on the FSDP training side (via external_lib) to also
    register the HF NavilaForCausalLM under AutoModelForCausalLM."""
    from transformers import AutoConfig, AutoProcessor
    try:
        AutoConfig.register("llava_llama", NaVILAConfig)
    except Exception:
        pass
    try:
        AutoProcessor.register(NaVILAConfig, NaVILAProcessor)
    except Exception:
        pass
    if with_model:
        from transformers import AutoModelForCausalLM
        from agent_system.models.navila_hf import NavilaForCausalLM
        try:
            AutoModelForCausalLM.register(NaVILAConfig, NavilaForCausalLM)
        except Exception:
            pass
