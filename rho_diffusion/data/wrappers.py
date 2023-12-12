from __future__ import annotations

from typing import Any
from typing import Callable

from torch import Tensor
from torchvision import transforms as t
from torchvision.datasets import CIFAR10
from torchvision.datasets import MNIST

from rho_diffusion.registry import registry


__all__ = ["CIFAR10Dataset", "MNISTDataset"]


@registry.register_dataset("CIFAR10Dataset")
class CIFAR10Dataset(CIFAR10):
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

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        data, label = super().__getitem__(index)
        return {"data": data, "label": label}

    @property
    def default_transforms(self) -> t.Compose:
        return t.Compose(
            [
                t.Resize((32, 32)),
                t.ToTensor(),  # Scales data into [0,1]
                t.Lambda(lambda t: (t * 2) - 1),  # Scale between [-1, 1]
            ],
        )


@registry.register_dataset("MNISTDataset")
class MNISTDataset(MNIST):
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

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        data, label = super().__getitem__(index)
        return {"data": data, "label": label}

    @property
    def default_transforms(self) -> t.Compose:
        return t.Compose(
            [
                t.Resize((32, 32)),
                t.ToTensor(),  # Scales data into [0,1]
                t.Lambda(lambda t: (t * 2) - 1),  # Scale between [-1, 1]
            ],
        )
