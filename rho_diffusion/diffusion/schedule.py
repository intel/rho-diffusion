from __future__ import annotations

import math
from abc import ABC
from copy import deepcopy

import torch
from torch import Tensor
from torch.nn.functional import pad

from rho_diffusion.registry import registry

__all__ = ["LinearSchedule", "CosineBetaSchedule", "SigmoidSchedule"]


class AbstractSchedule(ABC):
    @property
    def dtype(self) -> torch.dtype:
        dtype = getattr(self, "_dtype", None)
        if not dtype:
            dtype = torch.float32
        return dtype

    @dtype.setter
    def dtype(self, value: torch.dtype | None) -> None:
        if not value:
            value = torch.float32
        self._dtype = value

    @property
    def index(self) -> int:
        return self._index

    @index.setter
    def index(self, value: int) -> None:
        self._index = value

    @property
    def alpha_t(self) -> Tensor:
        return self._alpha_t.type(self.dtype)

    @alpha_t.setter
    def alpha_t(self, value: Tensor) -> None:
        self._alpha_t = value

    @property
    def beta_t(self) -> Tensor:
        return self._beta_t.type(self.dtype)

    @beta_t.setter
    def beta_t(self, value: Tensor) -> None:
        self._beta_t = value

    @property
    def alpha_bar_t(self) -> Tensor:
        return self._alpha_bar_t.type(self.dtype)

    @alpha_bar_t.setter
    def alpha_bar_t(self, value: Tensor) -> None:
        self._alpha_bar_t = value

    @property
    def offset_alpha_bar_t(self) -> Tensor:
        return pad(self.alpha_bar_t[:-1], (1, 0), value=1.0)

    @offset_alpha_bar_t.setter
    def offset_alpha_bar_t(self, value: Tensor) -> None:
        self._offset_alpha_bar_t = value

    @property
    def sigma_t(self) -> Tensor:
        return self._sigma_t.type(self.dtype)

    @sigma_t.setter
    def sigma_t(self, value: Tensor) -> None:
        self._sigma_t = value

    def state(self, index: int | None) -> dict[str, float]:
        if not index:
            index = self.index
        state_dict = {}
        for key in ["alpha_t", "beta_t", "alpha_bar_t", "sigma_t"]:
            state_dict[key] = getattr(self, key)[index]
        return state_dict

    @property
    def last_state(self) -> dict[str, float]:
        return self._last_state

    @last_state.setter
    def last_state(self, state: dict[str, float]):
        self._last_state = deepcopy(state)

    def reset(self) -> None:
        self.index = 0
        self.last_state = {}

    def step(self) -> None:
        if not hasattr(self, "_index"):
            self.reset()
        else:
            self.last_state = self.state
            self.index += 1

    def __getitem__(self, key: str) -> Tensor:
        return getattr(self, key)

    def __enter__(self):
        self.__old_dtype__ = self.dtype
        self.dtype = torch.float32

    def __exit__(self, *args, **kwargs):
        self.dtype = self.__old_dtype__


@registry.register_schedule("LinearSchedule")
class LinearSchedule(AbstractSchedule):
    def __init__(
        self,
        num_steps: int,
        beta_1: float = 1.0e-3,
        beta_T: float = 0.02,
        device="cpu",
    ) -> None:
        super().__init__()
        scale = 1000 / num_steps
        # work in a double precision context to prevent rounding errors
        with self:
            beta_t = torch.linspace(
                scale * beta_1,
                scale * beta_T,
                num_steps,
                device=device,
                dtype=torch.float64,
            )
            self.beta_t = beta_t.to(self.dtype)
            alpha_t = 1.0 - beta_t
            self.alpha_t = alpha_t.to(self.dtype)
            alpha_bar_t = alpha_t.cumprod(0)
            self.alpha_bar_t = alpha_bar_t.to(self.dtype)
            self.sigma_t = torch.sqrt(
                (1 - self.offset_alpha_bar_t) / (1 - alpha_bar_t) * beta_t,
            ).to(self.dtype)


@registry.register_schedule("CosineBetaSchedule")
class CosineBetaSchedule(AbstractSchedule):
    def __init__(self, num_steps: int, offset: float = 0.008, device="cpu") -> None:
        """
        Implements the cosine noise schedule described by Nichol and Dhariwal (2021).

        The intuition for this schedule is to provide a smooth ease into the
        signal to noise scheduling for the latent: effectively, last few steps
        of the forward process contain less noise than the linear schedule.

        Parameters
        ----------
        num_steps : int
            Number of steps to use for the schedule
        offset : float, optional
            Hyperparameter to prevent beta_t from becoming too small
            , by default 0.008, tuned by Nichol & Dhariwal.
        """
        super().__init__()
        with self:
            t = (
                torch.linspace(
                    0.0,
                    num_steps,
                    num_steps + 1,
                    dtype=torch.float64,
                    device=device,
                )
                / num_steps
            )
            alpha_bar_t = torch.cos((t + offset) / (1 + offset) * math.pi * 0.5).pow(
                2.0,
            )
            # normalize by the first value
            alpha_bar_t = alpha_bar_t.div(alpha_bar_t[0])
            self.alpha_bar_t = alpha_bar_t.to(self.dtype)
            self.alpha_bar_t[self.alpha_bar_t < 0] = 0  # fixes a NaN issue
            self.alpha_bar_t[self.alpha_bar_t > 1] = 1  # fixes a NaN issue
            beta_t = 1 - (alpha_bar_t / self.offset_alpha_bar_t)
            self.beta_t = beta_t.clip_(0.0001, 0.9999).to(self.dtype)
            self.alpha_t = (1 - beta_t).to(self.dtype)
            self.sigma_t = torch.sqrt(
                (1 - self.offset_alpha_bar_t) / (1 - alpha_bar_t) * beta_t,
            ).to(self.dtype)


@registry.register_schedule("SigmoidSchedule")
class SigmoidSchedule(AbstractSchedule):
    def __init__(self, num_steps: int, offset: float = 0.008) -> None:
        super().__init__()
        raise NotImplementedError(f"SigmoidSchedule is not yet implemented.")
