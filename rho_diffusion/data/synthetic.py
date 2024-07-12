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

from os import getenv
from pathlib import Path
from random import randint
from random import seed

import numpy as np
from h5py import File
from scipy.special import sph_harm
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from rho_diffusion.data.base import Density, MultiVariateDataset
from rho_diffusion.registry import registry
from rho_diffusion.utils import calculate_sha512_embedding
from rho_diffusion.data.parameter_space import DiscreteParameterSpace

"""
Blocks of code are adapted from the following article:

https://scipython.com/blog/visualizing-the-real-forms-of-the-spherical-harmonics/
"""


def make_spherical_grid(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> list[np.ndarray]:
    """
    Create a numerical grid for spherical coordinates ``theta`` and ``phi``
    from a 3D cartesian basis.

    Linearly spaced arrays for x,y,z are provided and mapped onto a meshgrid
    that is subsequently used to calculate the corresponding theta and phi
    values for each combination.

    Parameters
    ----------
    x : np.ndarray
        1D array with x coordinate values
    y : np.ndarray
        1D array with y coordinate values
    z : np.ndarray
        1D array with z coordinate values

    Returns
    -------
    list[np.ndarray]
        xyz, theta, and phi arrays; xyz corresponds to a stacked (4D)
        array containing the cartesian mesh grid, and theta and phi
        are corresponding values in spherical coordinates.
    """
    xg, yg, zg = np.meshgrid(x, y, z, indexing="xy")
    theta = np.arctan(np.sqrt(xg**2 + yg**2) / zg)
    phi = np.arctan(yg / xg)
    xyz = np.array([xg, yg, zg])
    return xyz, theta, phi


def compute_spherical_harmonic(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    m: int,
    l: int,
    normalize: bool = True,
) -> list[np.ndarray]:
    """
    Compute the spherical harmonic associated with order and degree m/n.

    Additional args allow you to specify the size of the grid to compute the
    spherical harmonics on, in spherical coordinates. Additionally, the
    optional ``normalized`` kwarg will return a normalized density.

    Parameters
    ----------
    m : int
        Order, or azimuthal quantum number
    l : int
        Degree, or orbital quantum number
    num_theta : int
        Number of elements in the theta grid
    num_phi : int
        Number of elements in the phi grid
    normalize : bool
        Whether to normalize density values to between [0,1],
        by default True

    Returns
    -------
    list[np.ndarray]
        _description_
    """
    xyz, theta, phi = make_spherical_grid(x, y, z)
    xg, yg, zg = xyz
    radial = np.sqrt(xg**2 + yg**2 + zg**2)
    solution = sph_harm(np.abs(m), l, theta, phi) * radial
    # if specified, normalize between 0,1
    if normalize:
        solution = (solution - solution.min()) / (solution.max() - solution.min())
    # if normalized, this doesn't really do much
    real_part = np.real(solution)
    return xyz, np.abs(solution), real_part


@registry.register_dataset("SphericalHarmonicDataset")
class SphericalHarmonicDataset(MultiVariateDataset):
    def __init__(
        self,
        max_l: int | None,
        h5_path: str | Path | None = None,
        length: int = 1000,
        random_seed: int | None = None,
        use_emb_as_labels: bool = True,
        **grid_kwargs,
    ):
        """
        Implements the base spherical harmonics density dataset.

        The dataset involves computing the spherical harmonics at
        (somewhat) arbitrary order and degree: the "data" returned
        is the solution of Ylm on an xyz grid.

        By providing ``h5_path`` as a path to an HDF5 file, this dataset
        class will also bypass random generation and directly retrieve
        samples from disk.

        Parameters
        ----------
        max_l : int | None
            _description_
        h5_path : Optional[str  |  Path  |  None], optional
            _description_, by default None
        length : int, optional
            _description_, by default 1000
        random_seed : Optional[int], optional
            _description_, by default None
        """

        parameter_space = DiscreteParameterSpace(
            param_dict={
                'l': list(range(0, max_l)),
                'm': list(range(-max_l, max_l))
            }
        )
        self.loaded_parameter_space = DiscreteParameterSpace()

        self.max_l = max_l
        self.random_seed = random_seed
        # configure default grid parameters
        grid_kwargs.setdefault("grid_el", 32)
        for key in ["x", "y", "z"]:
            grid_kwargs.setdefault(key, np.linspace(-2.0, 2.0, grid_kwargs["grid_el"]))
        self.grid_kwargs = grid_kwargs
        self.length = length
        self.h5_path = h5_path
        self.labels_emb_map = dict()

    @property
    def h5_path(self) -> Path | None:
        return self._h5_path

    @h5_path.setter
    def h5_path(self, value: str | Path | None) -> None:
        if isinstance(value, str):
            value = Path(value)
        if isinstance(value, Path):
            assert (
                value.exists()
            ), f"{value} passed as target HDF5 file but was not found."
        self._h5_path = value

    @property
    def h5_data(self) -> File:
        assert self.h5_path, f"Cannot retrieve HDF5 data because no path was provided."
        return File(str(self.h5_path), mode="r")

    @property
    def max_l(self) -> int:
        """
        Maximum value for the orbital angular momentum quantum number.

        Returns
        -------
        int
            Maximum value for l
        """
        return self._max_l

    @max_l.setter
    def max_l(self, value: int | None) -> None:
        if isinstance(value, int):
            assert value > 0, "Invalid maximum value of l > 0: {value}"
        self._max_l = value

    @property
    def random_seed(self) -> int:
        """
        Random seed used for l, m generation.

        The setter method will attempt to read a global seed value
        set by PyTorch Lightning initially, and fallback on a default
        seed for reproducibility.

        Returns
        -------
        int
            Integer value used for the random seed
        """
        return self._random_seed

    @random_seed.setter
    def random_seed(self, value: int | None) -> None:
        if not value:
            value = int(getenv("PL_GLOBAL_SEED", 1616))
        # set builtin RNG
        seed(value)
        self._random_seed = value

    @property
    def random_set(self) -> tuple[int]:
        """
        Generates a random 2-tuple of permissible l, m quantum numbers
        for data generation.

        Returns
        -------
        tuple[int]
            2-tuple containing l and m numbers
        """
        l = randint(0, self.max_l)  # noqa: E741, S311
        m = randint(-l, l)  # noqa: S311
        return (l, m)

    def __len__(self) -> int:
        if self.h5_path:
            with self.h5_data as h5_data:
                return len(h5_data["density"])
        else:
            return self.length

    def __getitem__(self, index: int) -> dict[str, Tensor | Density | dict[str, float]]:
        """
        Retrieve an example spherical harmonic entry.

        The behavior of this depends on how the dataset object was configured: if
        an ``h5_path`` was provided, we will load pre-computed samples from disk
        which may be desirable. Otherwise, each sample retrieved will be randomly
        generated over a configurable numerical grid, with random values for l
        and m provided each time.

        Parameters
        ----------
        index : int
            Index to retrieve a sample. If not loading from HDF5, this is
            not used.

        Returns
        -------
        dict[str, Tensor | Density | dict[str, float]]
            Dictionary containing a sample of data; the ``density`` key
            corresponds to a ``Density`` object (3D tensor), alongside
            the l and m values used to generate under the ``labels`` key.
        """
        if self.h5_path:
            with self.h5_data as h5_data:
                density = h5_data["density"][index]
                l = h5_data["l"][index]  # noqa: E741
                m = h5_data["m"][index]
        else:
            l, m = self.random_set
            grid_values = {key: self.grid_kwargs.get(key) for key in ["x", "y", "z"]}
            xyz, density, real_part = compute_spherical_harmonic(
                **grid_values,
                m=m,
                l=l,
            )
        c = {"l": l, "m": m}
        emb = calculate_sha512_embedding(c, l=256)
        self.labels_emb_map[emb] = c
        return (
            Density(density.astype(np.float32)).unsqueeze(0),
            emb,
        )

    def to_hdf5(self, h5_path: str | Path) -> None:
        """
        Serialize the dataset to an HDF5 file.

        This allows caching of the result so this doesn't have to be done at
        run time.

        Parameters
        ----------
        h5_path : str | Path
            Path to save the HDF5 data to. Will automatically add the
            ``.h5`` file extension if not included.
        """
        if isinstance(h5_path, str):
            h5_path = Path(h5_path).with_suffix(".h5")
        with File(str(h5_path), mode="x") as h5_file:
            all_data = [
                self.__getitem__(i)
                for i in tqdm(range(len(self)), desc="Samples generated")
            ]
            # stack up data into arrays for writing
            all_density = np.stack(data["density"] for data in all_data)
            all_l = np.array([data["labels"]["l"] for data in all_data])
            all_m = np.array([data["labels"]["m"] for data in all_data])
            # write data to disk
            h5_file["density"] = all_density
            h5_file["l"] = all_l
            h5_file["m"] = all_m
            h5_file.attrs["seed"] = self.random_seed

    @classmethod
    def from_hdf5(cls, h5_path: str | Path) -> SphericalHarmonicDataset:
        """
        Convenience method to explicitly load in the spherical harmonic
        dataset from an HDF5 file.

        Parameters
        ----------
        h5_path : str | Path
            Path to the HDF5 file
        """
        return cls(max_l=None, h5_path=h5_path)
