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

from itertools import product

import pytest
import torch

from rho_diffusion.models import vit


@pytest.mark.skip(reason="ViT not currently being used.")
@pytest.mark.parametrize(
    "data_dims, patch_size",
    [(1, 4), (1, 8), (1, 16), (2, 4), (2, 8), (2, 16), (3, 4), (3, 8), (3, 16)],
)
def test_patch_embedding(data_dims, patch_size):
    image_size = 64
    batch_size = 8
    num_channels = 1
    embedding_dim = 128
    num_patches = (image_size // patch_size) ** data_dims
    data = torch.rand(batch_size, num_channels, *(image_size,) * data_dims)
    embedder = vit.PatchEmbedding(
        num_channels,
        patch_size,
        embedding_dim,
        data_dims,
    ).eval()
    with torch.no_grad():
        output = embedder(data)
    assert output.size(0) == batch_size
    assert output.size(1) == num_patches
    assert output.size(-1) == embedding_dim


@pytest.mark.skip(reason="ViT not currently being used.")
@pytest.mark.parametrize(
    "image_size, num_channels, patch_size, data_dims",
    list(product([32, 64, 96], [1, 3], [4, 8, 16], [1, 2, 3])),
)
def test_vit_stack(image_size, num_channels, patch_size, data_dims):
    """Test a simple vision transformer stack on 1-3D data"""
    model = vit.VisionTransformer(
        patch_size=patch_size,
        input_shapes=(image_size,) * data_dims,
        num_channels=num_channels,
        embedding_dim=32,
        hidden_dim=64,
        activation="GELU",
        transformer_depth=1,
    ).eval()
    batch_size = 1
    input_data = torch.rand(batch_size, num_channels, *(image_size,) * data_dims)
    # generate timesteps for each batch entry
    t = torch.arange(batch_size)
    with torch.no_grad():
        output = model(input_data, t)
    assert output.shape == input_data.shape
