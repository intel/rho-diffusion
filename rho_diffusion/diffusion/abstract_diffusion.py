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


"""
Abstractions for diffusion noise scheduling
"""

import torch
from lightning import pytorch as pl
from matplotlib import pyplot as plt
from torch import nn
from torch import Tensor
from torch.optim import AdamW
from torch.optim import lr_scheduler
from torch.optim import Optimizer
from torchmetrics.image import PeakSignalNoiseRatio
from rho_diffusion.registry import registry
from rho_diffusion.diffusion import schedule
from collections.abc import Mapping
from typing import Any, Union
from inspect import getfullargspec
from logging import getLogger
import math 
from abc import ABC

from torchvision.utils import make_grid
from torchvision.utils import save_image


__all__ = ["AbstractDiffusionPipeline"]


class AbstractDiffusionPipeline(ABC, pl.LightningModule):
    def __init__(
        self,
        backbone: Union[str, type[nn.Module]],
        backbone_kwargs: dict[str, Any],
        schedule: schedule.AbstractSchedule,
        timesteps: Union[int, Tensor] = 1000,
        cond_fn: Union[str, type[nn.Module]] = None,
        cond_fn_kwargs: dict = None,
        optimizer: Union[str, type[nn.Module]] = None,
        opt_kwargs: Union[Mapping[str, Any], None] = {},
    ):
        super().__init__()
        if isinstance(backbone, str):
            backbone = registry.get("models", backbone)

        self.backbone = backbone(**backbone_kwargs)
        self.backbone_kwargs = backbone_kwargs
        if isinstance(cond_fn, str):
            cond_fn_class = registry.get('layers', cond_fn)
            self.backbone.cond_fn = cond_fn_class(**cond_fn_kwargs)
        if optimizer is None:
            optimizer = "AdamW"
        elif isinstance(optimizer, str):
            optimizer = registry.get("optimizers", optimizer)
        self.optimizer = optimizer
        self.schedule = schedule
        # hold a dictionary of torchmetrics objects
        self.metrics = nn.ModuleDict({"snr": PeakSignalNoiseRatio()})
        self.save_hyperparameters(
            {"model_kwargs": backbone_kwargs, "opt_kwargs": opt_kwargs},
        )
        self.timesteps = timesteps
        self._python_logger = getLogger(self.__class__.__name__)

    def configure_optimizers(self, mpi_world_size: int = 1) -> dict[Union[str, Optimizer, lr_scheduler._LRScheduler]]:
        """
        Set up the optimizer, and optionally, learning rate scheduler.

        Expects that ``opt_kwargs``

        Returns
        -------
        dict[str, Optimizer | lr_scheduler._LRScheduler]
            Dictionary containing the optimizer and learning rate scheduler
            as ``opt`` and ``lr_scheduler`` keys.

        Raises
        ------
        NameError
            _description_
        """
        opt_kwargs = self.hparams.opt_kwargs
        adam_spec = getfullargspec(AdamW)
        adam_kwargs = {}
        for key, value in zip(
            filter(lambda x: x not in ["self", "params"], adam_spec.args),
            adam_spec.defaults,
        ):
            if key not in opt_kwargs:
                adam_kwargs[key] = value
                opt_kwargs[key] = value
            elif key in opt_kwargs:
                adam_kwargs[key] = opt_kwargs[key]
        # instantiate the optimizer
        # opt = AdamW(self.parameters(), **adam_kwargs)
        # modify the learning rate according to the optional MPI world size
        opt_kwargs['lr'] = opt_kwargs['lr'] * math.sqrt(mpi_world_size)
        opt = self.optimizer(self.parameters(), **opt_kwargs)

        # create schedule
        if "lr_schedule" in opt_kwargs:
            schedule_params = opt_kwargs["lr_schedule"]
            schedule_class = getattr(lr_scheduler, schedule_params["class"], None)
            if not schedule_class:
                raise NameError(
                    f"Learning rate scheduler was missing, or invalid; passed: {schedule_class}",
                )
            # every other key/value pair in the dictionary is assumed
            # to be a scheduler parameter
            schedule_kwargs = {
                key: value for key, value in schedule_params.items() if key != "class"
            }
            scheduler = schedule_class(opt, **schedule_kwargs)
        else:
            # scheduler = lr_scheduler.OneCycleLR(
            #     opt,
            #     max_lr=opt_kwargs["lr"],
            #     total_steps=100000,
            # )
            scheduler = lr_scheduler.CosineAnnealingLR(
                opt,
                T_max=10,
                eta_min=opt_kwargs["lr"] / 10,
            )
        return {"optimizer": opt, "lr_scheduler": scheduler}

    @property
    def schedule(self) -> schedule.AbstractSchedule:
        return self._schedule

    @schedule.setter
    def schedule(self, _schedule) -> None:
        self._schedule = _schedule

    # @property
    # def timesteps(self) -> int:
    #     return len(self.schedule["beta_t"])

    def random_timesteps(self, num_steps: int) -> Tensor:
        steps = torch.randperm(self.timesteps)[:num_steps]
        return steps

    def reshape_timesteps(self, data: Tensor, t: Tensor) -> Tensor:
        """
        Generate the expected shape for timesteps based on input data.

        For images/volumes, this function will generate a view of ``t``,
        the timesteps, that is broadcast-ready.

        Parameters
        ----------
        data : Tensor
            Input data for noising
        t : Tensor
            Timestep tensor

        Returns
        -------
        Tensor
            The ``t`` tensor reshaped appropriately for broadcasting
        """
        #
        shape = (-1, *((1,) * (data.ndim - 1)))
        return t.view(shape)

    def get_schedule_parameters_at_time(
        self,
        data: Tensor,
        t: torch.LongTensor,
    ) -> dict[str, Tensor]:
        """
        Extracts out all of the relevant noise schedule parameters
        and formats them into a dictionary for access with the
        correct shape for broadcasting.

        Parameters
        ----------
        data : Tensor
            Example data to use for shape determination
        t : torch.LongTensor
            Timestep indices used to index schedule

        Returns
        -------
        dict[str, Tensor]
            Key/value pairs of each schedule parameter
        """
        result = {}
        for key in ["alpha_t", "beta_t", "alpha_bar_t", "sigma_t"]:
            parameter_slice = self.schedule[key].to(data.device)[t]
            result[key] = self.reshape_timesteps(data, parameter_slice)
        return result

    @property
    def state(self) -> dict[str, Tensor]:
        return self._state

    def forward_process(
        self,
        data: torch.Tensor,
        t: Union[Tensor, None] = None,
    ) -> dict[str, torch.Tensor]:
        """The forward diffusion process is to go from data to Gaussian"""
        ...

    def reverse_process(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """The reverse diffusion process is to go from noise to data"""
        latent = self.initial_latent(*args, **kwargs)
        with torch.no_grad():
            ...

    @staticmethod
    def make_image_grid(
        batched_image: Union[Tensor, list[Tensor]],
        filename: str = None,
    ) -> Tensor:
        """
        Utility function for taking a batch of images and creating a
        matplotlib ``Figure`` for display or logging.

        Parameters
        ----------
        batched_image : Tensor | list[Tensor]
            Batch of images, either as a stacked 4D tensor [BCHW] or
            as a list of 3D tensors [CHW].

        Returns
        -------
        plt.figure.Figure
            Matplotlib ``Figure`` object
        """
        if isinstance(batched_image, list):
            assert (
                batched_image[0].ndim == 3
            ), f"Individual images need to be 3D for gridding."
            batched_image = torch.stack(batched_image)
        # if tensor doesn't reside on CPU, move it there
        if batched_image.device != "cpu":
            batched_image = batched_image.cpu()
        assert batched_image.ndim == 4, f"Image grid needs to be 4D; dimensions BCHW."
        image_grid = make_grid(batched_image, normalize=True)
        # image_grid = image_grid.permute(1, 2, 0).numpy()  # reorder to HWC
        if filename is not None:
            save_image(image_grid, fp=filename)
        # fig, ax = plt.subplots()
        # ax.imshow(image_grid)
        # fig.tight_layout()
        return image_grid