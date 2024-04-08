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

from collections.abc import Mapping

import torch
from einops import rearrange
from torch import nn
from torch import Tensor

from rho_diffusion.models.common import SinusoidalPositionEmbedding
from rho_diffusion.registry import registry


class PatchEmbedding(nn.Module):
    """
    Layer that embeds an image into a sequence of patches.

    To simplify the process, we use a single convolution layer
    to generate ``embedding_dim`` number of channels, and emit
    an output corresponding to
    """

    def __init__(
        self,
        num_channels: int,
        patch_size: int,
        embedding_dim: int,
        data_dims: int,
    ) -> None:
        """
        Initialize a ``PatchEmbedding`` layer.

        This layer is used to ingest 1D-3D data, and generate a set
        of patches of a given patch size.

        Parameters
        ----------
        num_channels : int
            Number of channels in the input data.
        patch_size : int
            Size of data patches.
        embedding_dim : int
            Dimensionality of the resulting embedding, i.e.
            the output corresponds to ``[B N D]`` with ``N``
            number of patches, and ``D`` embedding size.
        data_dims : int
            Dimensionality of the data, i.e. 2 for images, 3 for voxels.
        """
        super().__init__()
        self.num_channels = num_channels  # this refers to the input image
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.data_dims = data_dims
        self._conv_type = registry.get("nn", f"Conv{self.data_dims}d")
        self.conv_shaper = self._conv_type(
            num_channels,
            embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    @property
    def data_dims(self) -> int:
        return self._data_dims

    @data_dims.setter
    def data_dims(self, value: int) -> None:
        assert 3 >= value > 0, f"data_dims must be between 1 and 3 for convolution."
        self._data_dims = value

    @property
    def input_shape(self) -> str:
        """Emits the expected input shape."""
        if self.data_dims == 3:
            return "n l h w d"
        elif self.data_dims == 2:
            return "n l h w"
        else:
            return "n l h"

    @property
    def output_shape(self) -> str:
        """Produces the expected output shape."""
        if self.data_dims == 3:
            return "n (h w d) l"
        elif self.data_dims == 2:
            return "n (h w) l"
        else:
            return "n h l"

    @property
    def ein_string(self) -> str:
        """Produces a string in Einstein notation for reshaping."""
        return f"{self.input_shape} -> {self.output_shape}"

    @property
    def stored_shape(self) -> tuple[int]:
        """Stores convolved shape sans batch size and channels"""
        return self._stored_shape

    @stored_shape.setter
    def stored_shape(self, value: tuple[int]) -> None:
        self._stored_shape = value

    def forward(self, data: Tensor) -> Tensor:
        conv_output = self.conv_shaper(data)
        conv_shape = tuple(conv_output.shape)
        # store the convolution shape for reshaping later
        self.stored_shape = conv_shape[2:]
        reshaped_output = rearrange(conv_output, self.ein_string)
        return reshaped_output


class AttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        activation: str | nn.Module = "GELU",
        time_dim: int = 128,
        **attn_kwargs,
    ):
        super().__init__()

        self.norm_1 = nn.LayerNorm(embed_dim)
        self.norm_2 = nn.LayerNorm(embed_dim)
        # set default values to map onto attention layer
        attn_kwargs.setdefault("batch_first", True)
        self.attention_layer = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout,
            **attn_kwargs,
        )

        # build up the linear projection layers
        activation = registry.get("activations", activation)
        self.linear_block = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        # time-embedding for diffusion process
        time_embedding = SinusoidalPositionEmbedding(time_dim)
        self.time_transform = nn.Sequential(
            time_embedding,
            nn.Linear(time_dim, embed_dim, bias=False),
            activation(),
        )

    def forward(self, data: Tensor, t: Tensor) -> dict[str, Tensor]:
        # embed diffusion time step, and add to data
        t_embed = self.time_transform(t).unsqueeze(1)
        embedded_data = data + t_embed
        # run through layer norm
        norm_data = self.norm_1(embedded_data)
        attn_o, attn_w = self.attention_layer(norm_data, norm_data, norm_data)
        # add attention output to the input
        attn_residual = norm_data + attn_o
        norm_attn_residual = self.norm_2(attn_residual)
        # add layer norm'd result to projection
        output = attn_residual + self.linear_block(norm_attn_residual)
        return {"output": output, "attn_weights": attn_w}


@registry.register_model("VisionTransformer")
class VisionTransformer(nn.Module):
    def __init__(
        self,
        patch_size: int,
        input_shapes: list[int],
        num_channels: int,
        embedding_dim: int,
        hidden_dim: int,
        activation: str | nn.Module,
        transformer_depth: int = 8,
        pos_embedding_dim: int = 128,
        time_embedding_dim: int = 128,
        max_seq_length: int = 20_000,
        dropout: float = 0.2,
        num_heads: int = 16,
        attention_kwargs: Mapping[str, any] | None = {},
    ) -> None:
        """
        Initialize a Vision Transformer model.

        At a high level, this model will embed 1D-3D data into
        a sequence of patches; a series of transformer models
        then operate on these patches as if they were tokens.

        This implementation is specialized for diffusion models
        as it also takes in a timestep from the diffusion process.

        Parameters
        ----------
        patch_size : int
            Size of the image patch; for now, the patch size
            is assumed to be the same for all dimensions but
            could be extended.
        input_shapes : list[int]
            Shapes of the input data, excluding the batch size
            and number of channels; for images this would be [H W],
            and for voxels, [H W D].
        num_channels : int
            Number of channels in the input data.
        embedding_dim : int
            Dimensionality of the embeddings; this is used for
            the patch embedding size between transformer layers.
        hidden_dim : int
            Hidden dimensionality used for the initial patch embedding
            size, and sizes within transformer layers.
        activation : str | nn.Module
            Nonlinearity to use globally; can be specified as either
            a string (e.g. ``GELU``, ``SiLU``).
        transformer_depth : int, optional
            Number of transformer layers to use, by default 8
        pos_embedding_dim : int, optional
            Dimensionality of the patch position encodings, by default 128
        time_embedding_dim : int, optional
            Dimensionality of the diffusion time step encoding, by default 128
        max_seq_length : int, optional
            Maximum sequence length; currently not being used, but could
            be passed into the positional encoding layer, by default 20_000
        dropout : float, optional
            Dropout probability between linear layers, by default 0.2
        num_heads : int, optional
            Number of attention heads, by default 16
        attention_kwargs : Mapping[str, any] | None, optional
            Kwargs to be passed into the creation of each ``nn.MultiHeadAttention``
            block, by default {}
        """
        super().__init__()
        # set the number dimensionality of the data, minus batch and channels
        data_dims = len(input_shapes)
        self.input_shapes = input_shapes
        self.patch_embedder = PatchEmbedding(
            num_channels,
            patch_size,
            embedding_dim,
            data_dims,
        )
        self.max_seq_length = max_seq_length
        self.transformer_blocks = nn.ModuleList(
            [
                AttentionBlock(
                    embedding_dim,
                    hidden_dim,
                    num_heads,
                    dropout,
                    activation,
                    time_embedding_dim,
                    **attention_kwargs,
                )
                for _ in range(transformer_depth)
            ],
        )
        self.output_projection = nn.Linear(embedding_dim, hidden_dim, bias=False)
        data_dims = self.data_dims
        # use transpose convolution to get back into shape
        conv_type = registry.get("nn", f"ConvTranspose{data_dims}d")
        self.output_conv = conv_type(
            hidden_dim,
            num_channels,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        activation = registry.get("activations", activation)
        # this is used to embed the patch positions
        self.pos_embedding = nn.Sequential(
            SinusoidalPositionEmbedding(pos_embedding_dim),
            nn.Linear(pos_embedding_dim, embedding_dim),
            activation(),
        )

    @property
    def input_shape_dim_names(self) -> str:
        keys = [key for _, key in zip(self.input_shapes, ["j", "k", "l"])]
        return keys

    @property
    def patch_size(self) -> int:
        return self.patch_embedder.patch_size

    @property
    def data_dims(self) -> int:
        return self.patch_embedder.data_dims

    @property
    def stored_shape(self) -> tuple[int]:
        return self.patch_embedder.stored_shape

    @property
    def stored_shape_ein(self) -> dict[str, int]:
        """Stores the initial convolution output shape"""
        return {key: value for key, value in zip(["j", "k", "l"], self.stored_shape)}

    def forward(self, input_data: Tensor, t: Tensor) -> Tensor:
        """
        Implements the forward method for the ViT architecture.

        This implementation includes three types of embedding: first,
        ``input_data`` is run through the patch embedder, which uses
        a convolution layer to embed the original data into a grid
        of patches; second, the positioning of each entry in the sequence
        (a la the original ViT) uses a sinusoidal embedding; third,
        the diffusion time step uses the same kind of sinusoidal embedding.

        The patch positional embedding is added to each patch, then follows
        a sequence of transformer layers that include the diffusion time
        embedding as well.

        Parameters
        ----------
        input_data : Tensor
            Input data, which could be 1D to 3D data. Assumes batch
            first.
        t : Tensor
            1D tensor of diffusion time step indices.

        Returns
        -------
        Tensor
            Output data, with the same shape as the input.
        """
        embedded_patches: Tensor = self.patch_embedder(input_data)
        # generate positional encodings for each patch
        seq_length = embedded_patches.size(1)
        patch_indices = torch.arange(seq_length, device=embedded_patches.device)
        pos_embedding = self.pos_embedding(patch_indices)
        # add positional embedding to the patches
        embedded_patches.add_(pos_embedding)
        for block in self.transformer_blocks:
            result = block(embedded_patches, t)
            embedded_patches = result["output"]
        # this projects the patches back into the same hidden dimension output
        # by the PatchEmbedder, with the goal to ultimately get to the same
        # shape as before
        embedded_patches = self.output_projection(embedded_patches)
        # reshape to resemble the original image shapes
        dim_names = " ".join(self.input_shape_dim_names)
        # use einsum notation to reshape the data; essentially reverse what
        # the original patch embedding convolution did
        reordered_patches = rearrange(
            embedded_patches,
            f"b ({dim_names}) c -> b c {dim_names}",
            **self.stored_shape_ein,
        )
        # use transpose convolution to recover the original shape
        output_data = self.output_conv(reordered_patches)
        return output_data
