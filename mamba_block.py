"""
Mamba block wrappers.

Adapted from the Mamba repository (Gu & Dao, 2023):
  https://github.com/state-spaces/mamba
Licensed under Apache-2.0.
"""

from typing import Optional

import torch.nn as nn
from torch import Tensor

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm
except ImportError:
    RMSNorm = None


class CrossBlock(nn.Module):
    def __init__(self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False):
        super().__init__()
        assert not fused_add_norm, \
            "fused_add_norm is not supported; CrossBlock uses the slow path."
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.norm_query = norm_cls(dim)

        if RMSNorm is not None:
            assert isinstance(self.norm, (nn.LayerNorm, RMSNorm)), \
                "Only LayerNorm and RMSNorm are supported."

    def forward(self, query, hidden_states: Tensor,
                residual: Optional[Tensor] = None, inference_params=None):
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        query = self.norm_query(query.to(dtype=self.norm_query.weight.dtype))
        hidden_states = self.mixer(query, hidden_states, inference_params=inference_params)
        return hidden_states, residual


class Block(nn.Module):
    def __init__(self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False):
        super().__init__()
        assert not fused_add_norm, \
            "fused_add_norm is not supported; Block uses the slow path."
        self.norm = norm_cls(dim)
        self.mixer = mixer_cls(dim)

        if RMSNorm is not None:
            assert isinstance(self.norm, (nn.LayerNorm, RMSNorm)), \
                "Only LayerNorm and RMSNorm are supported."

    def forward(self, hidden_states: Tensor,
                residual: Optional[Tensor] = None,
                inference_params=None, **mixer_kwargs):
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        hidden_states = self.mixer(hidden_states, inference_params=inference_params,
                                   **mixer_kwargs)
        return hidden_states, residual