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

from typing import Any
from typing import Callable

from torch import Tensor
from torchvision import transforms as t
from torchvision.datasets import CIFAR10
from torchvision.datasets import MNIST

from rho_diffusion.registry import registry
from rho_diffusion.data.base import UnivariateDataset
from rho_diffusion.data.parameter_space import DiscreteParameterSpace


__all__ = ["CIFAR10Dataset", "MNISTDataset"]


@registry.register_dataset("CIFAR10Dataset")
class CIFAR10Dataset(CIFAR10, UnivariateDataset):
    def __init__(
        self,
        root: str,
        transform: Callable[..., Any] | None = None,
        target_transform: Callable[..., Any] | None = None,
        **kwargs,
    ) -> None:
        kwargs.setdefault("download", True)
        kwargs.setdefault("train", True)
        if transform is None:
            transform = self.default_transforms
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            **kwargs,
        )

        self.parameter_space = DiscreteParameterSpace(
            param_dict={
                'labels': [0,1,2,3,4,5,6,7,8,9]
            }
        )

    # def __getitem__(self, index: int) -> dict[str, Tensor]:
    #     data, label = super().__getitem__(index)
    #     return {"data": data, "label": label}

    @property
    def default_transforms(self) -> t.Compose:
        return t.Compose(
            [
                # t.Resize((32, 32)),
                t.ToTensor(),  # Scales data into [0,1]
                t.Lambda(lambda t: (t * 2) - 1),  # Scale between [-1, 1]
            ],
        )


@registry.register_dataset("MNISTDataset")
class MNISTDataset(MNIST, UnivariateDataset):
    def __init__(
        self,
        root: str,
        transform: Callable[..., Any] | None = None,
        target_transform: Callable[..., Any] | None = None,
        **kwargs,
    ) -> None:
        kwargs.setdefault("download", True)
        kwargs.setdefault("train", True)
        if transform is None:
            transform = self.default_transforms
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            **kwargs,
        )

        self.parameter_space = DiscreteParameterSpace(
            param_dict={
                'labels': [0,1,2,3,4,5,6,7,8,9]
            }
        )

    # def __getitem__(self, index: int) -> dict[str, Tensor]:
    #     data, label = super().__getitem__(index)
    #     return {"data": data, "label": label}

    @property
    def default_transforms(self) -> t.Compose:
        return t.Compose(
            [
                t.Resize((32, 32)),
                t.ToTensor(),  # Scales data into [0,1]
                t.Lambda(lambda t: (t * 2) - 1),  # Scale between [-1, 1]
            ],
        )
