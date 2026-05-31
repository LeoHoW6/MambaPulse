"""
CrossMamba: a cross-attention variant of the Mamba selective state-space model.

Adapted from the Mamba repository (Gu & Dao, 2023):
  https://github.com/state-spaces/mamba
Licensed under Apache-2.0.
"""

import math

import torch
import torch.nn as nn
from einops import rearrange, repeat

from selective_scan_interface_ca import mamba_inner_fn

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None


class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2,
                                 bias=bias, **factory_kwargs)
        self.in_proj_q = nn.Linear(self.d_model, self.d_inner,
                                   bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state,
            bias=False, **factory_kwargs
        )
        self.q_proj = nn.Linear(
            self.d_inner, self.d_state, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner,
                                 bias=True, **factory_kwargs)

        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n", d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model,
                                  bias=bias, **factory_kwargs)

    def forward(self, query, hidden_states, inference_params=None):
        assert inference_params is None, \
            "CrossMamba only supports full-sequence forward."
        assert causal_conv1d_fn is not None, \
            "causal_conv1d is required; install causal-conv1d."

        _, seqlen, _ = hidden_states.shape

        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l", l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        qfc = rearrange(
            self.in_proj_q.weight @ rearrange(query, "b l d -> d (b l)"),
            "d (b l) -> b d l", l=seqlen,
        )
        if self.in_proj_q.bias is not None:
            qfc = qfc + rearrange(self.in_proj_q.bias.to(dtype=qfc.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())

        out = mamba_inner_fn(
            xz,
            qfc,
            self.conv1d.weight,
            self.conv1d.bias,
            self.x_proj.weight,
            self.q_proj.weight,
            self.dt_proj.weight,
            self.out_proj.weight,
            self.out_proj.bias,
            A,
            None,
            None,
            self.D.float(),
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )
        return out