from __future__ import annotations

import argparse
from pathlib import Path

import torch
from lightning import pytorch as pl
from torch.utils.data import DataLoader

from rho_diffusion import diffusion
from rho_diffusion.config import ExperimentConfig
from rho_diffusion.registry import registry
from rho_diffusion.utils import parameter_space_to_embeddings
import os 

use_ipex = False
try:
    from rho_diffusion import ipex
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
schedule = schedule_class(device=device, **config.noise_schedule.kwargs)

dset_class = registry.get("datasets", config.dataset.name)
dset = dset_class(**config.dataset.kwargs)

train_loader = DataLoader(
    dset,
    batch_size=config.training.batch_size,
    shuffle=True,
)

param_space_embeddings = parameter_space_to_embeddings(
    config.inference.parameter_space,
).to(device)

ddpm = diffusion.DDPM(
    backbone=config.model.name,
    backbone_kwargs=config.model.kwargs,
    schedule=schedule,
    loss_func=config.training.loss_fn,
    optimizer=config.optimizer.name,
    opt_kwargs=config.optimizer.kwargs,
    sample_every_n_epochs=config.training.sample_every_n_epochs,
    sampling_batch_size=len(param_space_embeddings),
    labels=param_space_embeddings,
)

# Load an existing model checkpoint (if given by command line argument explicity) to resume training 
if args.model_checkpoint_path is not None:
    model_checkpoint_path = args.model_checkpoint_path
    ddpm.backbone.load_state_dict(torch.load(model_checkpoint_path))

if use_ipex:
    # strategy = xpu.SingleXPUStrategy() if device == "xpu" else None
    # local_rank = int(os.environ.get('PMI_RANK', 0))
    # print('local_rank', local_rank)


    strategy = xpu.DDPXPUStrategy(process_group_backend='ccl',
                                  parallel_devices=[torch.device('xpu', 0), torch.device('xpu', 1)]) if device == "xpu" else None

    # strategy = xpu.DDPXPUStrategy(process_group_backend='ccl')

    trainer = pl.Trainer(
        strategy=strategy,
        min_epochs=config.training.min_epochs,
        max_epochs=config.training.max_epochs,
        callbacks=[ipex.IPEXCallback()],
        enable_checkpointing=False,
        # profiler='simple',
    )
else:
    trainer = pl.Trainer(
        min_epochs=config.training.min_epochs,
        max_epochs=config.training.max_epochs,
        enable_checkpointing=False,
        gpus=2
    )
trainer.fit(ddpm, train_dataloaders=train_loader)
