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
try:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input  # noqa: F401

    FLASH_ATTN_BERT_PADDING_AVAILABLE = True
except ImportError:
    from einops import rearrange  # noqa: F401
    from transformers.modeling_flash_attention_utils import _index_first_axis as index_first_axis  # noqa: F401
    from transformers.modeling_flash_attention_utils import _pad_input as pad_input  # noqa: F401
    from transformers.modeling_flash_attention_utils import _unpad_input as unpad_input  # noqa: F401

    FLASH_ATTN_BERT_PADDING_AVAILABLE = False

__all__ = ["index_first_axis", "pad_input", "rearrange", "unpad_input", "FLASH_ATTN_BERT_PADDING_AVAILABLE"]
