from __future__ import annotations

import torch
from torch import nn

from rho_diffusion.registry import registry

"""
Implements commonly used activation functions.
"""


__all__ = ["SymmetricLog"]


@registry.register_activation("SymmetricLog")
class SymmetricLog(nn.Module):

    """
    Implements the ``SymmetricLog`` activation as described in
    Cai et al., https://arxiv.org/abs/2111.15631

    The activation is asymptotic and provides gradients over
    a large range of possible values.
    """

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Apply the ``SymmetricLog`` activation function elementwise
        to the ``data``.

        Parameters
        ----------
        data : torch.Tensor
            Input PyTorch tensor to transform

        Returns
        -------
        torch.Tensor
            Resulting tensor with nonlinearity
        """
        tanx = data.tanh()
        return tanx * torch.log(data * tanx + 1)
