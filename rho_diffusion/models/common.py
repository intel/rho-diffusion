from __future__ import annotations

import torch
from torch import nn

from rho_diffusion.registry import registry


def sinosoidal_position_embedding(
    t: torch.Tensor,
    dim: int,
    wavelength: int = 10000,
) -> torch.Tensor:
    """
    t -> [sin(omega_1 * t), cos(omega_1 * t), sin(omega_2 * t), cos(omega_2 * t), .. sin(omega_(dim/2) * t, cos(omega_(dim/2) * t)]
    """
    assert dim % 2 == 0, "`dim` should be dividable by 2."
    device = t.device
    i = torch.arange(dim // 2, device=device)
    omega = torch.pow(wavelength, 2 * i / dim)
    pe = torch.empty(len(t), dim, device=device)
    pe[:, 2 * i] = torch.sin(t[:, None] / omega[None, :]).float()
    pe[:, 2 * i + 1] = torch.cos(t[:, None] / omega[None, :]).float()

    return pe


@registry.register_layer("SinusoidalPositionEmbedding")
class SinusoidalPositionEmbedding(nn.Module):
    """A Transformer-style position embedding generator.

    Given a time $t$ (integer), the position embedding is a vector of real numbers.
    This is analogue to converting $t$ to binary, in which every bit is
    encoded with either 0 or 1.
    The sinusoidal embedding is the contineous counterpart of binary
    embedding, and since it isdefined in the contineous domain, it
    has much larger capacity than binary embedding.

    Reference: Vaswani et al. (2017), section 3.5
    """

    def __init__(self, dim: int, wavelength: int = 10000) -> None:
        super().__init__()
        self.dim = dim
        self.wavelength = wavelength

    # def forward(self, t: torch.Tensor):
    #     """
    #     t -> [sin(omega_1 * t), cos(omega_1 * t), sin(omega_2 * t), cos(omega_2 * t), .. sin(omega_(dim/2) * t, cos(omega_(dim/2) * t)]
    #     """
    #     assert self.dim % 2 == 0, "`dim` should be dividable by 2."
    #     device = t.device
    #     i = torch.arange(self.dim // 2, device=device)
    #     omega = torch.pow(10000, 2 * i / self.dim)
    #     pe = torch.empty(len(t), self.dim, device=device)
    #     pe[:, 2 * i] = torch.sin(t[:, None] / omega[None, :]).float()
    #     pe[:, 2 * i + 1] = torch.cos(t[:, None] / omega[None, :]).float()

    #     return pe

    def forward(self, t: torch.Tensor):
        return sinosoidal_position_embedding(t, self.dim, self.wavelength)
