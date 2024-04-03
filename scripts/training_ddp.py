from __future__ import annotations

import argparse
from pathlib import Path

import torch
from lightning import pytorch as pl
from torch.utils.data import DataLoader

from rho_diffusion import diffusion
import intel_extension_for_pytorch as ipex 
from rho_diffusion import xpu
from rho_diffusion.config import ExperimentConfig
from rho_diffusion.registry import registry
from rho_diffusion.utils import parameter_space_to_embeddings
import os 
from tqdm import tqdm 

os.environ["PTI_ENABLE_COLLECTION"] = "0"

def setup_ddp():
    try:
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        init_method = 'tcp://127.0.0.1:29307'
        # init_method = 'file:///home/hpccai1/pytorch_tests/sync_file'
        # init_method = 'file:///tmp/sync_file'
        # init_method = 'env://'
        local_rank = int(os.environ.get('PMI_RANK', 0))
        mpi_world_size = int(os.environ.get('PMI_SIZE', -1))
        print('world size', mpi_world_size)
        torch.distributed.init_process_group(backend="ccl", init_method=init_method, world_size=mpi_world_size, rank=local_rank)
        return mpi_world_size, local_rank 
    except ValueError as e:
        print('No Intel MPI environment detected. Disabling distributed training...')
        return None, 0 

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
    "-e",
    "--epochs",
    dest="epochs",
    help="Number of epochs",
    type=int,
    default=10,
    required=False,
)
parser.add_argument(
    "-p",
    dest="model_checkpoint_path",
    help="path of the model checkpoint file to initialize the inference model",
    required=False,
)
parser.add_argument(
    "-l",
    "--learning_rate",
    dest="learning_rate",
    help="learning rate",
    type=float,
    default=0.01,
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

mpi_world_size, local_rank = setup_ddp()
if mpi_world_size is not None:
    device = torch.device("xpu:{}".format(local_rank))
else:
    device = torch.device("xpu")

# configure DDPM noise schedule and dataset
schedule_class = registry.get("schedules", config.noise_schedule.name)
schedule = schedule_class(device=device, **config.noise_schedule.kwargs)

dset_class = registry.get("datasets", config.dataset.name)
dset = dset_class(**config.dataset.kwargs)

if mpi_world_size is None:
    train_loader = DataLoader(
        dset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=8
    )
else:
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset=dset)
    train_loader = DataLoader(
        dset,
        batch_size=config.training.batch_size,
        sampler=train_sampler,
        num_workers=8
    )


param_space_embeddings = parameter_space_to_embeddings(
    config.inference.parameter_space,
).to(device)

model = diffusion.DDPM(
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
    model.backbone.load_state_dict(torch.load(model_checkpoint_path))



model = model.to(device)
if mpi_world_size is not None:
    model = torch.nn.parallel.DistributedDataParallel(model, 
                                                      device_ids=[local_rank], 
                                                      output_device=local_rank)
    optimizer = model.module.configure_optimizers(mpi_world_size)['optimizer']
else:
    optimizer = model.configure_optimizers()['optimizer']

# optimizer =registry.get("optimizers", optimizer)

# optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-5)

model, optimizer = ipex.optimize(model, optimizer=optimizer)

for epoch in range(args.epochs):
    if epoch > 0 and epoch < 3:
        os.environ["PTI_ENABLE_COLLECTION"] = "1"
    else:
        os.environ["PTI_ENABLE_COLLECTION"] = "0"
    with tqdm(total=len(dset), disable=(local_rank > 0)) as pbar:
        for step, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            
            # batch_size = data.size(0)
            # t =  torch.randperm(config.noise_schedule.kwargs.num_steps)[:batch_size].to(data.device)
            # x_data, noise = self.forward_process(data, t)
            if mpi_world_size is not None:
                loss = model.module.training_step([inputs, labels])
            else:
                loss = model.training_step([inputs, labels])
            loss.backward()
            optimizer.step()
            if mpi_world_size is not None:
                pbar.update(len(data[0]) * mpi_world_size)
                pbar.set_description('Epoch: %d/%d, NP: %d, step: %d, loss: %.5f' % (epoch + 1, args.epochs, mpi_world_size, step, loss.item()))

            else:
                pbar.update(len(data[0]))
                pbar.set_description('Epoch: %d/%d, NP: %d, step: %d, loss: %.5f' % (epoch + 1, args.epochs, 1, step, loss.item()))
    # if mpi_world_size is not None:
    #     model.module.current_epoch = epoch
    #     model.module.on_train_epoch_end()
    # else:
    #     model.current_epoch = epoch
    #     model.on_train_epoch_end()
