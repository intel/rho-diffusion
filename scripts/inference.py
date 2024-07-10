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

import argparse
import os
from pathlib import Path

import h5py
import torch
from lightning import pytorch as pl

from rho_diffusion import diffusion
from rho_diffusion.config import ExperimentConfig
from rho_diffusion.registry import registry
from rho_diffusion.utils import parameter_space_to_embeddings
from rho_diffusion.utils import plot_image_grid
from rho_diffusion.diffusion import DDPM 


try:
    import intel_extension_for_pytorch as ipex

    use_ipex = True
except ImportError:
    use_ipex = False

parser = argparse.ArgumentParser()
parser.add_argument(
    "json_config",
    type=Path,
    help="Path to JSON config file; see examples folder.",
)
parser.add_argument(
    "-p",
    dest="model_checkpoint_path",
    type=Path,
    help="path of the model checkpoint file to initialize the inference model",
    required=False,
)
parser.add_argument(
    "-d",
    "--device",
    dest="device",
    help="compute device [cpu/xpu]",
    type=str,
    default="xpu",
    required=False,
)
parser.add_argument(
    "-n",
    dest="n_samples",
    type=int,
    help="number of samples to generate",
    default=10,
)
parser.add_argument(
    "-f",
    dest="forced_overwrite",
    action="store_true",
    help="force the script to overwrite existing inference output data file",
    default=False,
)

args = parser.parse_args()

config = ExperimentConfig.from_json(args.json_config)


if args.device is not None:
    # override the `device` setting in the config file
    device = args.device
else:
    device = config.inference.device

pl.seed_everything(config.inference.seed)

generated_data_fn = config.inference.cache_file
if os.path.isfile(generated_data_fn) and args.forced_overwrite is False:
    if config.inference.plot_output_file is not None:
        print(
            "Found a cache copy of the generated data: %s. Will plot this file. If you wish to plot newly generated data, please use the `-f` argument to overwrite."
            % generated_data_fn,
        )
        with h5py.File(generated_data_fn, "r") as h5f:
            pred_images = torch.tensor(h5f["data"][()])
            plot_image_grid(
                pred_images,
                transpose=False,
                filename=config.inference.plot_output_file,
            )
else:
    schedule_class = registry.get("schedules", config.noise_schedule.name)
    schedule = schedule_class(device=device, **config.noise_schedule.kwargs)
    parameter_space = registry.get("datasets", config.dataset.name).parameter_space

    # if config.dataset.kwargs["use_emb_as_labels"]:
    #     param_space_labels = parameter_space_to_embeddings(
    #         config.inference.parameter_space,
    #     ).to(device)
    # else:
    #     param_space_labels = torch.tensor(config.inference.parameter_space).to(device)

    # inference_batch_size = min(len(param_space_labels), args.n_samples)

    model = diffusion.GaussianDiffusionPipeline(
    # model = diffusion.DDPM(
        # backbone=config.model.name,
        # backbone_kwargs=config.model.kwargs,
        # schedule=schedule,
        # loss_func=config.training.loss_fn,
        # optimizer=config.optimizer.name,
        # opt_kwargs=config.optimizer.kwargs,
        # sample_every_n_epochs=config.training.sample_every_n_epochs,
        # sampling_batch_size=inference_batch_size,
        # labels=param_space_labels[:inference_batch_size],


        backbone=config.model.name,
        backbone_kwargs=config.model.kwargs,
        schedule=schedule,
        loss_func=config.training.loss_fn,
        timesteps=config.noise_schedule.kwargs['num_steps'],
        cond_fn=config.model.kwargs['cond_fn'],
        cond_fn_kwargs={'parameter_space': parameter_space, 'embedding_dim': 128},
        optimizer=config.optimizer.name,
        opt_kwargs=config.optimizer.kwargs,
        sample_every_n_epochs=config.training.sample_every_n_epochs,
        sampling_batch_size=args.n_samples,
        sample_parameter_space=config.inference.parameter_space
    )

    if args.model_checkpoint_path is not None:
        model_checkpoint_path = args.model_checkpoint_path
    else:
        model_checkpoint_path = config.inference.checkpoint
    model.backbone.load_state_dict(torch.load(model_checkpoint_path))

    # model = DDPM.load_from_checkpoint(model_checkpoint_path)
    model.eval()
    model.to(device)
    print(model)

    # import sys; sys.exit(0)
    if use_ipex:
        model = ipex.optimize(model)
        ipex.xpu.synchronize()

    # pred_images = model.p_sample()
    pred_images = model.generate(parameter_space=config.inference.parameter_space, random=False)

    with h5py.File(generated_data_fn, "w") as h5f:
        h5f["data"] = pred_images.cpu().numpy()

    if config.inference.plot_output_file is not None:
        plot_image_grid(
            pred_images,
            transpose=False,
            filename=config.inference.plot_output_file,
        )
