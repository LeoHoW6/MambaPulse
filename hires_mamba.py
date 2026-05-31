import math
import torch
import torch.nn as nn
from functools import partial

from timm.layers import DropPath
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_block import Block

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm
except ImportError:
    RMSNorm = None

if RMSNorm is None:
    class RMSNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-5, **kwargs):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(hidden_size))

        def forward(self, x, **kwargs):
            variance = x.float().pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            return (x * self.weight).to(dtype=x.dtype)


def build_2d_sincos_pe(H: int, W: int, d_model: int,
                       device=None, dtype=torch.float32) -> torch.Tensor:
    assert d_model % 4 == 0, f"d_model must be divisible by 4, got {d_model}"
    d_half = d_model // 2

    y = torch.arange(H, device=device, dtype=dtype).unsqueeze(1)
    x = torch.arange(W, device=device, dtype=dtype).unsqueeze(0)
    div = torch.exp(
        torch.arange(0, d_half, 2, device=device, dtype=dtype)
        * (-math.log(10000.0) / d_half)
    )

    pe = torch.zeros(H, W, d_model, device=device, dtype=dtype)
    sin_y = torch.sin(y.unsqueeze(2) * div)
    cos_y = torch.cos(y.unsqueeze(2) * div)
    pe[:, :, 0:d_half:2] = sin_y.expand(-1, W, -1)
    pe[:, :, 1:d_half:2] = cos_y.expand(-1, W, -1)
    sin_x = torch.sin(x.unsqueeze(2) * div)
    cos_x = torch.cos(x.unsqueeze(2) * div)
    pe[:, :, d_half::2] = sin_x.expand(H, -1, -1)
    pe[:, :, d_half + 1::2] = cos_x.expand(H, -1, -1)

    return pe.reshape(H * W, d_model)


class HiResMambaBranch(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        d_model: int = 128,
        out_channels: int = 256,
        num_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        patch_size: int = 4,
        max_hw: int = 200,
        drop_path: float = 0.1,
        layer_scale_init: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.max_hw = max_hw

        self.patch_embed = nn.Conv2d(
            in_channels, d_model,
            kernel_size=patch_size, stride=patch_size, bias=True,
        )
        self.input_norm = nn.LayerNorm(d_model)

        pe = build_2d_sincos_pe(max_hw, max_hw, d_model)
        self.register_buffer("pos_embed_2d", pe.reshape(max_hw, max_hw, d_model),
                             persistent=False)

        dpr = [x.item() for x in torch.linspace(0, drop_path, num_layers)]
        self.fw_blocks = nn.ModuleList()
        self.bw_blocks = nn.ModuleList()
        self.fw_drop_paths = nn.ModuleList()
        self.bw_drop_paths = nn.ModuleList()
        self.fuse_projs = nn.ModuleList()
        for i in range(num_layers):
            self.fw_blocks.append(
                Block(
                    d_model,
                    mixer_cls=partial(Mamba, d_state=d_state,
                                      d_conv=d_conv, expand=expand),
                    norm_cls=partial(RMSNorm, eps=1e-5),
                    fused_add_norm=False,
                )
            )
            self.bw_blocks.append(
                Block(
                    d_model,
                    mixer_cls=partial(Mamba, d_state=d_state,
                                      d_conv=d_conv, expand=expand),
                    norm_cls=partial(RMSNorm, eps=1e-5),
                    fused_add_norm=False,
                )
            )
            self.fw_drop_paths.append(
                DropPath(dpr[i]) if dpr[i] > 0. else nn.Identity()
            )
            self.bw_drop_paths.append(
                DropPath(dpr[i]) if dpr[i] > 0. else nn.Identity()
            )
            self.fuse_projs.append(nn.Linear(d_model * 2, d_model, bias=False))

        self.layer_scale = nn.Parameter(
            layer_scale_init * torch.ones(d_model)
        )

        self.out_norm = RMSNorm(d_model, eps=1e-5)
        self.out_proj = nn.Sequential(
            nn.Conv2d(d_model, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.patch_embed.weight,
                                mode='fan_out', nonlinearity='linear')
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)
        for fuse in self.fuse_projs:
            nn.init.xavier_uniform_(fuse.weight, gain=1.0 / math.sqrt(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        x = self.patch_embed(x)
        _, _, Hp, Wp = x.shape
        N = Hp * Wp

        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.input_norm(x)

        if Hp > self.max_hw or Wp > self.max_hw:
            pe = build_2d_sincos_pe(Hp, Wp, self.d_model,
                                    device=x.device, dtype=x.dtype)
        else:
            pe = self.pos_embed_2d[:Hp, :Wp, :].reshape(N, self.d_model)
            pe = pe.to(dtype=x.dtype)
        x = x + pe.unsqueeze(0)

        residual = None
        for fw, bw, fw_dp, bw_dp, fuse in zip(
            self.fw_blocks, self.bw_blocks,
            self.fw_drop_paths, self.bw_drop_paths,
            self.fuse_projs,
        ):
            fw_out, fw_res = fw(x, residual=residual)
            fw_out = fw_dp(fw_out)

            bw_in = x.flip([1])
            bw_res_in = residual.flip([1]) if residual is not None else None
            bw_out, bw_res = bw(bw_in, residual=bw_res_in)
            bw_out = bw_dp(bw_out)
            bw_out = bw_out.flip([1])
            bw_res = bw_res.flip([1])

            x = fuse(torch.cat([fw_out, bw_out], dim=-1))
            residual = (fw_res + bw_res) * 0.5

        if residual is not None:
            x = x + residual
        x = self.out_norm(x)

        x = x * self.layer_scale

        x = x.transpose(1, 2).reshape(B, self.d_model, Hp, Wp).contiguous()
        x = self.out_proj(x)
        return x


class CNNStem(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2,
                      kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU(),
            nn.Conv2d(out_channels // 2, out_channels,
                      kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.stem(x)