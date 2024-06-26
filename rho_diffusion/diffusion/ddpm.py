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
from __future__ import annotations

from abc import ABC
from collections.abc import Iterable
from collections.abc import Mapping
from inspect import getfullargspec
from logging import getLogger
from typing import Any

import torch
from lightning import pytorch as pl
from matplotlib import pyplot as plt
from torch import nn
from torch import Tensor
from torch.optim import AdamW
from torch.optim import lr_scheduler
from torch.optim import Optimizer
from torchmetrics.image import PeakSignalNoiseRatio
from torchvision.utils import make_grid
from torchvision.utils import save_image
from tqdm import tqdm
import math

from rho_diffusion.diffusion import schedule
from rho_diffusion.registry import registry
from rho_diffusion.utils import save_model_checkpoint


__all__ = ["DDPM"]


class AbstractDiffusionPipeline(ABC, pl.LightningModule):
    def __init__(
        self,
        backbone: str | type[nn.Module],
        backbone_kwargs: dict[str, Any],
        schedule: schedule.AbstractSchedule,
        timesteps: int | Tensor = 1000,
        optimizer: str | type[nn.Module] = None,
        opt_kwargs: Mapping[str, Any] | None = {},
    ):
        super().__init__()
        if isinstance(backbone, str):
            backbone = registry.get("models", backbone)
        self.backbone = backbone(**backbone_kwargs)
        self.backbone_kwargs = backbone_kwargs
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

    def configure_optimizers(self, mpi_world_size: int = 1) -> dict[str, Optimizer | lr_scheduler._LRScheduler]:
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

    @property
    def num_timesteps(self) -> int:
        return len(self.schedule["beta_t"])

    def random_timesteps(self, num_steps: int) -> Tensor:
        steps = torch.randperm(self.num_timesteps)[:num_steps]
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
        t: Tensor | None = None,
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
        batched_image: Tensor | list[Tensor],
        filename: str = None,
    ) -> plt.figure.Figure:
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


class DDPM(AbstractDiffusionPipeline):
    def __init__(
        self,
        backbone: str | type[nn.Module],
        backbone_kwargs: dict[str, Any],
        schedule: schedule.AbstractSchedule,
        loss_func: str | type[nn.Module] | nn.Module,
        timesteps: int | Tensor = 1000,
        optimizer: str | type[nn.Module] = None,
        opt_kwargs: Mapping[str, Any] | None = {},
        labels: Tensor = None,
        t_checkpoints=None,
        sampling_batch_size=10,
        sample_every_n_epochs=5,
        save_weights_every_n_epochs=10,
    ):
        super().__init__(
            backbone,
            backbone_kwargs,
            schedule,
            timesteps,
            optimizer,
            opt_kwargs,
        )
        if isinstance(loss_func, str):
            loss_func = registry.get("nn", loss_func)
        if isinstance(loss_func, type):
            loss_func = loss_func()
        self.loss_func = loss_func
        self.labels = labels
        self.t_checkpoints = t_checkpoints
        self.sampling_batch_size = sampling_batch_size
        self.sample_every_n_epochs = sample_every_n_epochs
        self.save_weights_every_n_epochs = save_weights_every_n_epochs

    def noise(self, data: Tensor) -> Tensor:
        return torch.randn_like(data)

    def forward_process(
        self,
        data: Tensor,
        t: Tensor | None = None,
    ) -> list[Tensor]:
        batch_size = data.size(0)
        # send data type to schedule to type cast consistently
        self.schedule.dtype = data.dtype
        if t is None:
            t = self.random_timesteps(batch_size)
    
        # reshapes the time indices so that alpha bar values
        # will be in the right shape for broadcasting
        if t.ndim == 1:
            t = self.reshape_timesteps(data, t)

        # t.to(data.device)
        self.schedule.to(data.device)
        alpha_bar_t = self.schedule["alpha_bar_t"].to(data.device)[t].to(data.device)
        noise = self.noise(data)
        # add noise to data

        posterior_mean = alpha_bar_t.sqrt() * data
        posterior_var = (1 - alpha_bar_t).sqrt() * noise

        x_data = posterior_mean + posterior_var
        return [x_data, noise]

    @torch.inference_mode()
    def reverse_process(
        self,
        x_T: Tensor,
        conditions: Tensor = None,
        t_checkpoints: list | Tensor = None,
    ) -> dict[str, Tensor]:
        """
        Implements the reverse process: given a noisy image, iterately denoise the image.
        Algorithm 2 in Ho et al. (2020).

        Args:
            x_T (Tensor): The source (noised) tensor to denoise
            conditions (Tensor, optional): The conditioning info for the sampling process
            t_checkpoints (list | Tensor, optional): _description_. Defaults to None.

        Returns:
            dict[str, Tensor]: A dict consists of the denoised/generated tensors,
                               and a tensor that consists of the intermediate steps
                               if `t_checkpoints` is not None
        """
        schedule = self.schedule
        batch_size = x_T.size(0)
        # send data type to schedule to type cast consistently
        self.schedule.dtype = x_T.dtype

        if t_checkpoints is not None:
            num_checkpoints = len(t_checkpoints)
            # make a dummy tensor to get it into the right shape
            dummy = torch.ones_like(x_T, dtype=x_T.dtype).unsqueeze(1)
            dummy = torch.repeat_interleave(dummy, num_checkpoints, 1)
            # dummy serves as a template for the right shape
            denoised_img_buff = torch.zeros_like(dummy)
        else:
            denoised_img_buff = None

        denoise_steps = len(schedule["alpha_bar_t"])
        steps_per_ckpt = denoise_steps // 10

        x_t = torch.randn_like(x_T)
        steps = torch.arange(denoise_steps - 1, -1, -1)

        if conditions is not None:
            if isinstance(conditions, int):
                cc = torch.full(
                    (batch_size,),
                    fill_value=conditions,
                    device=x_t.device,
                    dtype=torch.long,
                )
            elif isinstance(conditions, str) and conditions == "auto":
                cc = torch.randint(0, 10, (batch_size,), device=x_T.device).long()
            elif isinstance(conditions, torch.Tensor):
                cc = conditions
            elif isinstance(conditions, list):
                cc = torch.tensor(conditions).to(x_t.device)
        else:
            cc = None

        # iterate through timesteps
        t_idx = 0
        if self.global_rank == 0:
            steps = tqdm(steps, desc="Reverse diffusion process...")
        for t in steps:
            if t > 1:
                z = torch.randn_like(x_t)
            else:
                z = torch.zeros_like(x_t)
            tt = torch.full(
                (batch_size,),
                fill_value=t,
                device=self.device,
                dtype=torch.long,
            )
            # get out noise schedule parameters in correct shape
            sch_params = self.get_schedule_parameters_at_time(x_T, tt)
            # predict image noise at given timestep
            pred_noise = self.backbone(x_t, tt, cc)
            if t > 0:
                x_t = (1 / sch_params["alpha_t"].sqrt()) * (
                    x_t
                    - (sch_params["beta_t"] / (1 - sch_params["alpha_bar_t"]).sqrt())
                    * pred_noise
                ) + 0.8 * torch.sqrt(sch_params["beta_t"]) * z

                x_t = torch.clamp(x_t, -1, 1)

            # add image to buffer
            if (
                denoised_img_buff is not None
                and t % steps_per_ckpt == 0
                and t_idx < num_checkpoints
            ):
                denoised_img_buff[:, t_idx] = x_t
                t_idx += 1

        return {"buffer": denoised_img_buff, "denoised": x_t}

    def training_step(self, batch: Iterable[Any]) -> float:
        """
        Given data, the training objective is to train the backbone
        to be epsilon-theta, i.e. supervised training to predict
        noise.

        Parameters
        ----------
        batch : Iterable[Any]
            Batched data and label pairs, either as a list or
            dictionary.

        Returns
        -------
        float
            Value of the loss to perform backprop with
        """

        if isinstance(batch, list):
            # assume it's a pair of image and labels like MNIST
            data, labels = batch
        elif isinstance(batch, dict):
            data = batch.get("data")
            labels = batch.get("label")
        else:
            # no label
            data = batch
            labels = None

        self.data_shape = data.shape
        self.data_dtype = data.dtype
        # for training, we learn via the forward process
        batch_size = data.size(0)
        t = self.random_timesteps(batch_size).to(data.device)

        x_data, noise = self.forward_process(data, t)

        if x_data.isnan().sum() > 0:
            print("Error: Noised data contains NaNs. Check your noise scheduler.")
            import sys

            sys.exit(0)

        # run epsilon_theta for noise prediction
        if labels is not None:
            pred_noise = self.backbone(x_data, t, labels)
        else:
            pred_noise = self.backbone(x_data, t)

        loss = self.loss_func(pred_noise, noise)
        # as a gauge of how noisy the images are,
        # we can compute the peak SNR between noise-free and noisy data
        for key, metric in self.metrics.items():
            value = metric(x_data, data)
            self.log(f"train_{key}", metric)
        # log the noise loss
        self.log(f"train_loss", loss)
        return loss

    def forward(self, batch: Iterable[Any]) -> Tensor:
        """A compatibility function to allow this module to be used as a standard PyTorch Module.

        Args:
            batch (Iterable[Any]): _description_

        Returns:
            Any: _description_
        """
        return self.backbone(batch)

    def on_train_epoch_end(self) -> None:
        """
        Check if there is a need to run certain tasks.
        """
        if (
            self.current_epoch > 0 and self.sample_every_n_epochs > 0 
            and self.current_epoch % self.sample_every_n_epochs == 0
        ):
            self.eval()
            self.p_sample()

        if (
            self.current_epoch > 0 and self.save_weights_every_n_epochs > 0
            and self.current_epoch % self.save_weights_every_n_epochs == 0
        ):
            self.eval()
            self.save_model_weights()

    def p_sample(self):
        if hasattr(self, "data_shape"):
            # infer the data size from `self.data`
            sampling_data_shape = [int(x) for x in self.data_shape]
            sampling_data_shape[0] = self.sampling_batch_size
        else:
            # Construct the shape for inference
            bs = self.sampling_batch_size
            channels = self.backbone_kwargs["out_channels"]
            shape = self.backbone_kwargs["data_shape"]
            sampling_data_shape = torch.Size([bs, channels] + shape)
            self.data_dtype = torch.float32

        sample_data = torch.zeros(
            sampling_data_shape,
            dtype=self.data_dtype,
            device=self.device,
        )

        results = self.reverse_process(
            x_T=sample_data,
            conditions=self.labels,
            t_checkpoints=self.t_checkpoints,
        )
        figure = self.make_image_grid(
            results["denoised"],
            filename="output_%d.png" % self.current_epoch,
        )
        # step = self.trainer.global_step
        # rank = self.trainer.global_rank
        # self.logger.experiment.log_figure(
        #     self.logger.run_id,
        #     figure,
        #     f"images/train_step{step}_rank{rank}_images.png",
        # )
        return figure

    def save_model_weights(self):
        print("saving model checkpoints...")
        save_model_checkpoint(self.backbone, "model.pth")

    def validation_step(self, batch: Iterable[Any]) -> float:
        """
        Skip validation. Idential to a train step.
        """
        return 0
