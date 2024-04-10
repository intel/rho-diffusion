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

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import Tensor


class Density(Tensor):
    """
    Abstracts the concept of "density" as a volumetric tensor.

    The concept behind this representation is to provide some level of
    convenience to go between sampled points and the underlying density
    on a grid. Later on, we can also add functionality that manipulates
    densities, such as renormalization, etc. that might help with toying
    around with different modeling behaviors, featurizations, etc.

    For the meantime, we interpret N-dimensional densities this way:
    1D -> density along a 1D grid
    2D -> multiple channels of 1D densities
    3D -> multiple channels of 2D densities
    """

    def __init__(self, data: ArrayLike, *grid, indexing: str = "ij") -> None:
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        elif isinstance(data, torch.tensor):
            # do nothing if it's already a tensor
            pass
        else:
            data = torch.as_tensor(data)
        self.data = data
        self.indexing = indexing
        if len(grid) == 0:
            # create our own generic grid
            grid = [torch.arange(s) for s in data.shape]
        self.grid = grid

    @property
    def grid(self) -> tuple[torch.tensor]:
        return self._grid

    @grid.setter
    def grid(self, grid_tensors: tuple(ArrayLike)) -> None:
        grid = []
        for index, value in enumerate(grid_tensors):
            if not isinstance(value, Tensor):
                value = torch.as_tensor(value)
            if len(value) != self.size(index):
                raise ValueError(
                    f"Length of grid along dimension {index} does not match data; expected {self.size(index)}, got {len(value)}",
                )
            grid.append(value)
        self._grid = torch.meshgrid(*grid, indexing=self.indexing)

    def normalize(self, inplace: bool = False) -> Density | None:
        """
        Normalize the density by dividing by the sum/integral.

        If ``inplace`` is ``True``, the operation is done without returning
        a new ``Density`` object, otherwise we will create a clone.

        Parameters
        ----------
        inplace : bool, optional
            If True, data contained in this ``Density`` is renormalized without
            creating another tensor, by default False

        Returns
        -------
        Union[Density, None]
            If ``inplace`` is True, None is returned. Otherwise,
            create a clone of the current density and normalize
            the new object.
        """
        if inplace:
            self.data.div_(self.data.sum())
            return None
        copy_density = self.data.clone()
        return Density(copy_density.div_(copy_density.sum()))

    def marginalize(self, dims: list[int]) -> Density:
        ...

    def sample(self, num_points: int) -> Tensor:
        """
        Sample ``num_points`` from this density.

        The resulting tensor would be shape ``[N, D]``, with ``N`` samples
        and ``D`` dimensionality.

        TODO make this work with ``torch.multinomial``

        Parameters
        ----------
        num_points : int
            Number of points to sample from the density.

        Returns
        -------
        Tensor
            Sampled points from the density.
        """
        raise NotImplementedError("Sampling from density is not yet implemented.")

    def __deepcopy__(self, memo):
        return super().__deepcopy__(memo)
