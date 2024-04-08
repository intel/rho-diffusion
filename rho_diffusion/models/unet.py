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

from rho_diffusion.models.common import SinusoidalPositionEmbedding
from rho_diffusion.registry import registry

__all__ = ["UNetBlock2d", "UNetBlock3d", "UNet"]


class AbstractUNetBlock(nn.Module):
    """convolution building block for the U-net with time embedding
    injection and optional resnet architecture.

    This implements the abstract case; concrete classes fill in
    the expected private attributes depending on the dimensionality
    of the expected data.
    """

    __conv_class__ = None
    __transpose_class__ = None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embedding_dim: int,
        is_up: bool = False,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 8,
        activation: str | nn.Module = "GELU",
        residual: bool = False,
    ) -> None:
        super().__init__()
        self.time_embedding_readout = nn.Linear(time_embedding_dim, out_channels)
        if is_up:
            self.conv1 = self.__conv_class__(
                in_channels=2 * in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
            self.conv2 = self.__transpose_class__(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        else:
            self.conv1 = self.__conv_class__(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
            self.conv2 = self.__conv_class__(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )

        if residual:
            if is_up:
                self.residual_conv = self.__transpose_class__(
                    in_channels=2 * in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            else:
                self.residual_conv = self.__conv_class__(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
        else:
            self.residual_conv = None

        if groups == 0:
            self.norm = None
        else:
            self.norm = nn.GroupNorm(groups, out_channels)

        if isinstance(activation, str):
            activation = registry.get("activations", activation)()
        self.activation = activation

    def forward(self, x, t):
        time_pe = self.time_embedding_readout(
            t,
        )  # TODO: test whether it's helpful to warp `time_pe` with an activation function
        h = self.activation(self.conv1(x))
        h = self.activation(self.conv2(h))

        if self.residual_conv is not None:
            h = h + self.residual_conv(x)

        # To save model parameters, directly add the
        # time embedding to the hidden represenation (as oppose of concat)
        h = (
            h + time_pe[(...,) + (None,) * 2]
        )  # add two more dimenstion to `time_pe` to match the shape of `h`

        if self.norm is not None:
            h = self.norm(h)

        return self.activation(h)


@registry.register_layer("UNetBlock2d")
class UNetBlock2d(AbstractUNetBlock):
    __conv_class__ = nn.Conv2d
    __transpose_class__ = nn.ConvTranspose2d


@registry.register_layer("UNetBlock3d")
class UNetBlock3d(AbstractUNetBlock):
    __conv_class__ = nn.Conv3d
    __transpose_class__ = nn.ConvTranspose3d


@registry.register_model("UNet")
class UNet(nn.Module):
    def __init__(
        self,
        block_type: str | AbstractUNetBlock,
        input_channels: int,
        down_channels: list[int] = [64, 128, 256],
        up_channels: list[int] = [256, 128, 64],
        time_embedding_dim: int = 32,
        kernel_size: int = 3,
        padding: int = 1,
        activation: str | nn.Module = "ReLU",
        residual: bool = True,
    ) -> None:
        super().__init__()
        # map name of block to registry entry
        if isinstance(block_type, str):
            block_type = registry.get("layers", block_type)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(time_embedding_dim),
            nn.Linear(time_embedding_dim, time_embedding_dim),
        )  # TODO: check whether it is helpful to add an activation function here

        # generate input and output layers depending on if we expect
        # volumetric and image data
        if block_type == UNetBlock3d:
            # for volumetric data
            layer_type = nn.Conv3d
        else:
            layer_type = nn.Conv2d
        self.input_conv = layer_type(
            in_channels=input_channels,
            out_channels=down_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.output_conv = layer_type(
            up_channels[-1],
            input_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        # create the down sampling blocks
        self.downsample = nn.ModuleList(
            [
                block_type(
                    in_channels=down_channels[i],
                    out_channels=down_channels[i + 1],
                    time_embedding_dim=time_embedding_dim,
                    is_up=False,
                    kernel_size=kernel_size,
                    padding=padding,
                    activation=activation,
                    residual=residual,
                )
                for i in range(len(down_channels) - 1)
            ],
        )
        # create the upsampling blocks
        self.upsample = nn.ModuleList(
            [
                block_type(
                    in_channels=up_channels[i],
                    out_channels=up_channels[i + 1],
                    time_embedding_dim=time_embedding_dim,
                    is_up=True,
                    kernel_size=kernel_size,
                    padding=padding,
                    activation=activation,
                    residual=residual,
                )
                for i in range(len(up_channels) - 1)
            ],
        )
        self.block_type = block_type

    @property
    def expected_dim(self) -> int:
        if self.block_type == UNetBlock2d:
            return 3
        else:
            return 4

    def forward(self, data: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for UNet, modified for diffusion processes.

        In addition to the ``data`` tensor that holds an image, volume, etc.,
        the ``t`` tensor represents integer counts of the time step that gets
        mapped onto a sinusoidal embedding space like for transformers.

        Parameters
        ----------
        data : torch.Tensor
            Tensor containing data to be transformed; i.e. x_t in papers
        t : torch.Tensor
            Time steps for data samples

        Returns
        -------
        torch.Tensor
            Output image/volume from UNet
        """
        time_pe = self.time_mlp(t)
        x = self.input_conv(data)

        # hidden states for skipped connections
        residual_h = []
        for down_block in self.downsample:
            x = down_block(x, time_pe)
            residual_h.append(x)
        for up_block in self.upsample:
            x = up_block(torch.cat((x, residual_h.pop()), dim=1), time_pe)

        return self.output_conv(x)
