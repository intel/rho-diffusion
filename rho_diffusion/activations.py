# Copyright (C) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#
#
# SPDX-License-Identifier: Apache-2.


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
