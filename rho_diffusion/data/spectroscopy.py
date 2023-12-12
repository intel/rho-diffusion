from __future__ import annotations

from functools import cached_property
from pathlib import Path
from random import gauss

import numpy as np
from einops import rearrange
from einops import reduce
from h5py import File
from torch import Tensor
from torch.utils.data import Dataset

from rho_diffusion.registry import registry


@registry.register_dataset("SpectroscopyDataset")
class SpectroscopyDataset(Dataset):
    """
    Implements a dataset class for a rotational spectrum.

    This acts more or less as a testbed for 1D diffusion modeling,
    with the idea that you could potentially use the trained model
    to generate rotational spectrum on a set grid, on the fly
    conditioned on the parameters used to generate the spectrum.
    """

    def __init__(
        self,
        h5_path: str | Path,
        min_freq: float | None = None,
        max_freq: float | None = None,
        grid_size: int = 50_000,
        linewidth: float | tuple[float, float] | None = None,
    ) -> None:
        super().__init__()
        self.h5_path = h5_path
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.grid_size = grid_size
        self.linewidth = linewidth

    @property
    def linewidth(self) -> float:
        linewidth = self._linewidth
        if not isinstance(linewidth, float):
            if isinstance(linewidth, tuple):
                mu, sigma = linewidth
            else:
                mu, sigma = 1.0, 0.1
            return np.abs(gauss(mu, sigma))
        else:
            return linewidth

    @linewidth.setter
    def linewidth(self, value: float | tuple[float, float] | None) -> None:
        if isinstance(value, tuple):
            assert len(value) == 2, f"Expected two-tuple for linewidth specification."
        self._linewidth = value

    @property
    def h5_data(self) -> File:
        return File(str(self.h5_path), mode="r")

    @property
    def h5_path(self) -> Path:
        return self._h5_path

    @h5_path.setter
    def h5_path(self, value: str | Path) -> None:
        if isinstance(value, str):
            value = Path(value)
        assert value.exists(), f"Target HDF5 not found; passed {value}."
        self._h5_path = value

    @cached_property
    def __len__(self) -> int:
        with self.h5_data as h5_data:
            return len(h5_data)

    @cached_property
    def frequency_grid(self) -> np.ndarray:
        # add 20% on either side as defaults
        min_freq = getattr(self, "min_freq")
        if not min_freq:
            min_freq = 1000
        max_freq = getattr(self, "max_freq")
        if not max_freq:
            max_freq = 32000
        freq_grid = np.linspace(min_freq, max_freq, self.grid_size, dtype=np.float32)
        return freq_grid

    def __getitem__(self, index) -> dict[str, Tensor]:
        with self.h5_data as h5_data:
            group = h5_data[str(index)]
            data = {
                key: np.array(value).astype(np.float32) for key, value in group.items()
            }
        # generate some width from normal distribution
        width = np.abs(gauss(1.0, 0.1))
        centers, intensities = data["transitions"]
        # clip intensities as some of them may have underflowed
        intensities = np.clip(intensities, -10.0, -2.0)
        # intensities are stored in log10
        lineprofile = self.simulate_lineprofile(
            self.frequency_grid,
            centers,
            10**intensities,
            width,
        )
        max_int = 10 ** intensities.max()
        lineprofile /= lineprofile.max()
        # just return the dictionary from HDF5
        del data["transitions"]
        # add a channel dimension for consistency with other data
        data["spectrum"] = rearrange(Tensor(lineprofile), "f -> () f")
        data["max_int"] = Tensor([max_int])
        for key in data.keys():
            _d = data.get(key)
            if not isinstance(_d, Tensor):
                data[key] = Tensor(_d).float()
        return data

    @staticmethod
    def simulate_lineprofile(
        frequency_grid: np.ndarray,
        centers: np.ndarray,
        intensities: np.ndarray,
        width: float | np.ndarray,
    ) -> np.ndarray:
        """
        Simulates the spectrum given a set of transition centers
        and intensities.

        This model assumes Gaussian lineshapes for the sake of
        simplicity and speed.

        Parameters
        ----------
        frequency_grid : np.ndarray
            NumPy 1D array containing frequency channels.
        centers : np.ndarray
            NumPy 1D array containing transition center frequencies.
        intensities : np.ndarray
            NumPy 1D array containing transition intensities, assuming
            log10 values.
        width : float | np.ndarray
            Width(s) of the lineshape; either a single value is
            used uniformly for all transitions (i.e. a single
            body of gas) or can also accept different widths for
            each transition.

        Returns
        -------
        np.ndarray
            NumPy 1D array containing the spectrum simulated on
            the ``frequency_grid``.
        """
        if isinstance(width, float):
            width = np.array([width])
        min_freq, max_freq = frequency_grid.min(), frequency_grid.max()
        # mask out transitions that are outside our frequency range
        mask = np.all([centers <= max_freq, min_freq <= centers], axis=0)
        lineprofile = intensities[mask, None] * np.exp(
            -((frequency_grid[None, :] - centers[mask, None]) ** 2.0)
            / (2 * width[:, None] ** 2.0),
        )
        # sum up all of the components, remaining should be intensity
        # at each frequency channel
        lineprofile = reduce(lineprofile, "c f -> f", reduction="sum")
        return lineprofile
