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
