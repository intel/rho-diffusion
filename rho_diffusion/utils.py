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

import hashlib
import itertools
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.utils import save_image
from typing import Union

use_ipex = False
try:
    import intel_extension_for_pytorch as ipex
    import oneccl_bindings_for_pytorch

    use_ipex = True
except ImportError:
    use_ipex = False


def ddp_setup(port=29600):
    mpi_world_size = int(os.environ.get("PMI_SIZE", -1))
    mpi_rank = int(os.environ.get("PMI_RANK", -1))
    mpi_local_rank = int(os.environ.get("MPI_LOCALRANKID", -1))
    if mpi_world_size > 0:
        os.environ["RANK"] = str(mpi_rank)
        os.environ["WORLD_SIZE"] = str(mpi_world_size)
    else:
        # set the default rank and world size to 0 and 1
        os.environ["RANK"] = str(os.environ.get("RANK", 0))
        os.environ["WORLD_SIZE"] = str(os.environ.get("WORLD_SIZE", 1))
    os.environ["MASTER_ADDR"] = "127.0.0.1"  # your master address
    os.environ["MASTER_PORT"] = "29600"  # your master port
    # Initialize the process group with ccl backend
    init_method = "tcp://127.0.0.1:%d" % port
    if mpi_world_size > 1:
        if use_ipex:
            dist.init_process_group(
                backend="ccl",
                init_method=init_method,
                world_size=mpi_world_size,
                rank=mpi_rank,
            )
        else:
            dist.init_process_group(
                backend="nccl",
                init_method=init_method,
                world_size=mpi_world_size,
                rank=mpi_rank,
            )
    else:
        # make it possible to run it with a single process, with and without using `mpirun`
        mpi_world_size = 1
        mpi_rank = 0
    if mpi_rank == 0:
        print("Running on %d workers" % mpi_world_size)
    return mpi_world_size, mpi_rank, mpi_local_rank


def plot_image_grid(img_grid, transpose=False, filename=None):
    # td = torch.tensor(img_grid.swapaxes(0, 1)).clamp(-1, 1)
    td = img_grid.clamp(-1, 1)

    if transpose:
        td = td.swapaxes(0, 1)
        td = td.reshape(td.shape[0] * td.shape[1], td.shape[2], *td.shape[3:])
        rendered_img_grid = make_grid(
            td,
            nrow=img_grid.shape[0],
            padding=2,
            pad_value=1,
            normalize=True,
        ).cpu()
    else:
        td = td.reshape(td.shape[0] * td.shape[1], td.shape[2], *td.shape[3:])
        rendered_img_grid = make_grid(
            td,
            nrow=img_grid.shape[1],
            padding=2,
            pad_value=1,
            normalize=True,
        ).cpu()
    # fig, axs = plt.subplots(nrows=img_grid.shape[0], ncols=img_grid.shape[1])
    # for i in range(img_grid.shape[0]):
    #     for j in range(img_grid.shape[1]):
    #         axs[i, j].imshow(img_grid[i, j])
    #         axs[i, j].axis('off')
    if filename is not None:
        save_image(rendered_img_grid, fp=filename)
    else:
        return rendered_img_grid


def plot_tensor_images(tensor_img_grid, dim, filename=None):
    tensor_img_grid = tensor_img_grid.cpu()
    if dim == 2:
        reverse_transforms = transforms.Compose(
            [
                transforms.Lambda(lambda t: (t + 1) / 2),
                transforms.Lambda(lambda t: t.permute(0, 1, 3, 4, 2)),  # CHW to HWC
                transforms.Lambda(lambda t: t * 255.0),
                transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
                #     transforms.ToPILImage(),
            ],
        )
        img_grid = reverse_transforms(tensor_img_grid)
        plt.figure(figsize=(img_grid.shape[0] * 3, img_grid.shape[1] * 3))
        fig, axs = plt.subplots(nrows=img_grid.shape[0], ncols=img_grid.shape[1])
        for i in range(img_grid.shape[0]):
            for j in range(img_grid.shape[1]):
                axs[i, j].imshow(img_grid[i, j])
                axs[i, j].axis("off")
    elif dim == 3:
        reverse_transforms = transforms.Compose(
            [
                transforms.Lambda(lambda t: (t + 1) / 2),
                transforms.Lambda(lambda t: t.squeeze()),
                transforms.Lambda(lambda t: t * 255.0),
                transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
                #     transforms.ToPILImage(),
            ],
        )
        img_grid = reverse_transforms(tensor_img_grid)
        fig = plt.figure(figsize=(img_grid.shape[0] * 3, img_grid.shape[1] * 3))
        for i in range(img_grid.shape[0]):
            for j in range(img_grid.shape[1]):
                nth = i * img_grid.shape[0] + j + 1
                ax = fig.add_subplot(
                    img_grid.shape[0],
                    img_grid.shape[1],
                    nth,
                    projection="3d",
                )
                ax.voxels(img_grid[i, j])
                ax.axis("off")
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()


def save_model_checkpoint(model, filename):
    torch.save(model.state_dict(), filename)


def calculate_sha512_embedding(d: dict, l: int = 128) -> np.array:
    h = hashlib.sha512(json.dumps(d, sort_keys=True).encode()).hexdigest()
    # convert to ASCII numbers, then repeat to make the embedding having the desired length,
    # then normalized to 128, which is the total number of ASCII characters to get a float vector
    return torch.tensor(
        np.array(h, "c").view(np.uint8).repeat(l // 128) / 128,
        dtype=torch.float32,
    )


def vector_to_embeddings(v, keyname):
    emb = []
    for i in range(len(v)):
        emb.append(calculate_sha512_embedding({keyname: int(v[i])}))
    return torch.tensor(np.array(emb))


def parameter_space_to_embeddings(param_dict: dict) -> torch.Tensor:
    """Converts a parameter space into a list of embeddings.

    Example:
        `param_dict = {'m': [1,2], 'n': [3,4,5]}` will result in a list of embeddings corresponding to
        `{'m': 1, 'n': 3}`,
        `{'m': 2, 'n': 3}`,
        `{'m': 1, 'n': 4}`,
        `{'m': 2, 'n': 4}`,
        `{'m': 1, 'n': 5}`,
        `{'m': 2, 'n': 5}`,


    Args:
        param_dict (dict): The parameter space

    Returns:
        torch.tensor: The resulting tensor of embeddings.
    """
    keys, values = zip(*param_dict.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    emb = []
    for i in range(len(combinations)):
        emb.append(calculate_sha512_embedding(combinations[i]))
    return torch.stack(emb)

def sample_from_discrete_parameter_space(param_dict: dict, batch_size: int, random=True, device=None) -> torch.Tensor:
    keys, values = zip(*param_dict.items())
    combinations = torch.tensor([v for v in itertools.product(*values)], device=device)
    if random:
        idx = torch.randint(low=0, high=combinations.shape[0], size=(batch_size,), device=device)
    else:
        idx = torch.arange(start=0, end=batch_size, step=1, device=device)
    return combinations[idx]


def number_cast_dict(input_dict: dict) -> dict:
    """
    Casts dictionary values into floats/integers as allowed.
    """
    new_dict = {}

    def _type_cast(_input):
        try:
            _input = float(_input)
            if _input.is_integer():
                _input = int(_input)
        except ValueError:
            pass
        return _input

    for key, value in input_dict.items():
        if isinstance(value, list):
            value = [_type_cast(v) for v in value]
        else:
            value = _type_cast(value)
        new_dict[key] = value
    return new_dict

def right_pad_dims_to(x: torch.tensor, t: torch.tensor) -> torch.tensor:
    """
    Pads `t` with empty dimensions to the number of dimensions `x` has. If `t` does not have fewer dimensions than `x`
        it is returned without change.
    """
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

