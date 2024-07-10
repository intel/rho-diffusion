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
from pathlib import Path

import torch
from lightning import pytorch as pl
from torch.utils.data import DataLoader

from rho_diffusion import diffusion
from rho_diffusion.lightning_progress_bar import TrainingProgressBar
from rho_diffusion.config import ExperimentConfig
from rho_diffusion.registry import registry
from rho_diffusion.models.conditioning import MultiEmbeddings
from rho_diffusion.utils import parameter_space_to_embeddings
import os 

use_ipex = False
try:
    # from rho_diffusion import ipex
    from rho_diffusion import xpu
    use_ipex = True
except ImportError:
    use_ipex = False
    print('Cannot find Intel extension for PyTorch. The model cannot run on Intel GPUs.')
    torch.backends.cuda.matmul.allow_tf32 = False 
    torch.backends.cudnn.allow_tf32 = False

parser = argparse.ArgumentParser()
parser.add_argument(
    "json_config",
    help="Path to a JSON configuration file; see examples folder.",
    type=Path,
)
parser.add_argument(
    "-d",
    "--device",
    dest="device",
    help="Compute device [cpu/xpu]",
    type=str,
    default="xpu",
    required=False,
)
parser.add_argument(
    "-p",
    dest="model_checkpoint_path",
    help="path of the model checkpoint file to initialize the inference model",
    required=False,
)
args = parser.parse_args()

config = ExperimentConfig.from_json(args.json_config)
pl.seed_everything(config.training.seed)

if args.device is not None:
    # override the `device` setting in the config file
    device = args.device
else:
    device = config.training.device
print(config)

# configure DDPM noise schedule and dataset
schedule_class = registry.get("schedules", config.noise_schedule.name)
schedule = schedule_class(**config.noise_schedule.kwargs)

dset_class = registry.get("datasets", config.dataset.name)
dset = dset_class(**config.dataset.kwargs)

train_loader = DataLoader(
    dset,
    batch_size=config.training.batch_size,
    shuffle=True,
)

# param_space_embeddings = parameter_space_to_embeddings(
#     config.inference.parameter_space,
# ).to(device)

# cond_fn = MultiEmbeddings(dset.parameter_space, embedding_dim=128).to(device)

ddpm = diffusion.GaussianDiffusionPipeline(
    backbone=config.model.name,
    backbone_kwargs=config.model.kwargs,
    schedule=schedule,
    loss_func=config.training.loss_fn,
    timesteps=config.noise_schedule.kwargs['num_steps'],
    cond_fn=config.model.kwargs['cond_fn'],
    cond_fn_kwargs={'parameter_space': dset.parameter_space, 'embedding_dim': 128},
    optimizer=config.optimizer.name,
    opt_kwargs=config.optimizer.kwargs,
    sample_every_n_epochs=config.training.sample_every_n_epochs,
    save_checkpoint_every_n_epochs=config.training.save_checkpoint_every_n_epochs,
    sampling_batch_size=10,
    sample_parameter_space=config.inference.parameter_space
)

# Load an existing model checkpoint (if given by command line argument explicity) to resume training 
if args.model_checkpoint_path is not None:
    model_checkpoint_path = args.model_checkpoint_path
    ddpm.backbone.load_state_dict(torch.load(model_checkpoint_path))

if use_ipex:
    # strategy = xpu.SingleXPUStrategy() if device == "xpu" else None
    strategy = xpu.DDPXPUStrategy(process_group_backend='ccl')

    trainer = pl.Trainer(
        strategy='xpu_ddp',
        # strategy='deepspeed_stage_1',
        accelerator='xpu',
        devices=strategy.cluster_environment.world_size(),
        min_epochs=config.training.min_epochs,
        max_epochs=config.training.max_epochs,
        callbacks=[TrainingProgressBar()],
        enable_checkpointing=False,
        # profiler='simple',
    )
else:
    trainer = pl.Trainer(
        min_epochs=config.training.min_epochs,
        max_epochs=config.training.max_epochs,
        enable_checkpointing=False,
        accelerator='gpu', 
        devices=1
    )

trainer.fit(ddpm, train_dataloaders=train_loader)
trainer.save_checkpoint("model.ckpt")
