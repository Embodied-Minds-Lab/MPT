import torch
import torch.nn as nn
from torch import Tensor


class DownSampleBlock(nn.Module):
    """NaVILA's 2x2 spatial downsample (base_projector.DownSampleBlock.flat_square): reshape the
    patch grid, pad odd sides with zeros, fold 2x2 neighborhoods into the channel dim -> sequence
    length /4, channels *4. For siglip 729 patches -> 196 tokens/frame, channels 1152 -> 4608."""

    def forward(self, x: Tensor) -> Tensor:
        n, seq, c = x.shape
        h = w = int(seq ** 0.5)
        x = x.reshape(n, h, w, c)
        if w % 2 == 1:
            x = torch.cat([x, torch.zeros((n, 1, x.shape[2], c), dtype=x.dtype, device=x.device)], dim=1)
        if h % 2 == 1:
            x = torch.cat([x, torch.zeros((n, x.shape[1], 1, c), dtype=x.dtype, device=x.device)], dim=2)
        x = x.view(n, x.shape[1], x.shape[2] // 2, c * 2)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(n, x.shape[1], x.shape[2] // 2, c * 4)
        return x.reshape(n, -1, c * 4)


class NaVILAProjector(nn.Module):
    """mm_projector_type == 'mlp_downsample'. Layer order matches NaVILA's checkpoint tensor names
    layers.{1,2,4}: DownSample(0) -> LayerNorm(1) -> Linear(2) -> GELU(3) -> Linear(4)."""

    def __init__(self, mm_hidden: int, hidden: int, dtype=None):
        super().__init__()
        self.layers = nn.Sequential(
            DownSampleBlock(),                       # 0 (no params)
            nn.LayerNorm(mm_hidden * 4),             # 1  weight/bias [mm_hidden*4]
            nn.Linear(mm_hidden * 4, hidden),        # 2  [hidden, mm_hidden*4]
            nn.GELU(),                               # 3 (no params)
            nn.Linear(hidden, hidden),               # 4  [hidden, hidden]
        )
        if dtype is not None:
            self.layers.type(dtype)

    def forward(self, x: Tensor) -> Tensor:
        # Match the projector's own dtype WITHOUT next(self.parameters()), which raises
        # StopIteration under FSDP (submodule params are flattened/sharded). layers[1] is the
        # LayerNorm, whose weight is gathered during the forward in both FSDP and sglang.
        w = self.layers[1].weight
        return self.layers(x.to(device=w.device, dtype=w.dtype))
