from __future__ import annotations

from rho_diffusion.data.deep_galaxy import DeepGalaxyDataset
from rho_diffusion.data.spectroscopy import SpectroscopyDataset
from rho_diffusion.data.synthetic import SphericalHarmonicDataset
from rho_diffusion.data.wrappers import CIFAR10Dataset
from rho_diffusion.data.wrappers import MNISTDataset

__all__ = [
    "SphericalHarmonicDataset",
    "MNISTDataset",
    "CIFAR10Dataset",
    "SpectroscopyDataset",
    "DeepGalaxyDataset",
]
