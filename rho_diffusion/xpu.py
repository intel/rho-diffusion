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
An XPU (Intel GPU) plugin for Lightning.

This code is ported from MatSciML (MIT License): https://github.com/IntelLabs/matsciml/blob/main/matsciml/lightning/xpu.py
Copyright (C) 2023 Intel Corporation
"""


from __future__ import annotations

import os
from socket import gethostbyname
from typing import List, Any, Callable
from datetime import timedelta
import torch
import intel_extension_for_pytorch as ipex
from torch.nn.parallel.distributed import DistributedDataParallel
from lightning.pytorch.accelerators import Accelerator
from lightning.pytorch.accelerators import AcceleratorRegistry
from lightning.pytorch.plugins.environments import LightningEnvironment
from lightning.pytorch.plugins.precision import MixedPrecisionPlugin 
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.strategies import SingleDeviceStrategy, StrategyRegistry
from lightning.pytorch.plugins.io import CheckpointIO
from lightning.pytorch.plugins.environments import ClusterEnvironment
from lightning.fabric.plugins.collectives.torch_collective import default_pg_timeout
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_only, rank_zero_warn
# from lightning.lite.utilities.seed import reset_seed
from lightning.pytorch.utilities.seed import isolate_rng
from torch import distributed as dist
from typing import Optional
import logging 
import random
from contextlib import nullcontext
from typing_extensions import override
from lightning.fabric.utilities.distributed import _distributed_is_initialized


class IntelMPIEnvironment(LightningEnvironment):
    """
    This environment specializes in the use of Intel MPI for distributed
    multiworker instances. The key assumptions for using this environment
    are:
    1. The use of Intel MPI
    2. The launch script utilizes PyTorch Lightning abstractions
    3. The launch script is used via `mpiexec -n -ppn ... python train.py
    The main motivation behind this environment is two-fold: to keep the
    `pl.Trainer` functionality, while maintaining the ability to work with
    NUMA bindings (e.g. via `-map-by numa`) to ensure optimal CPU/memory
    utilization.
    """

    def __init__(
        self,
        main_address: str | None = None,
        main_port: int | None = None,
    ) -> None:
        super().__init__()
        self.main_address = main_address
        if main_port:
            self._main_port = main_port

    def world_size(self) -> int:
        if 'PMI_SIZE' in os.environ:
            return int(os.environ["PMI_SIZE"])
        elif 'OMPI_COMM_WORLD_SIZE' in os.environ:
            return int(os.environ['OMPI_COMM_WORLD_SIZE'])
        elif 'WORLD_SIZE' in os.environ:
            return int(os.environ['WORLD_SIZE'])
        else:
            return 1  # not invoked with MPI
        

    def local_rank(self) -> int:
        if 'MPI_LOCALRANKID' in os.environ:
            return int(os.environ["MPI_LOCALRANKID"])
        elif 'OMPI_COMM_WORLD_RANK' in os.environ:
            return int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        elif 'LOCAL_RANK' in os.environ:
            return int(os.environ['LOCAL_RANK'])
        else:
            return 0

    def global_rank(self) -> int:
        if 'MPI_LOCALRANKID' in os.environ:
            return int(os.environ["MPI_LOCALRANKID"])
        elif 'OMPI_COMM_WORLD_RANK' in os.environ:
            return int(os.environ['OMPI_COMM_WORLD_RANK'])
        elif 'RANK' in os.environ:
            return int(os.environ['RANK'])
        else:
            return 0

    @property
    def main_address(self) -> str:
        return self._main_address

    @main_address.setter
    def main_address(self, value: str | None) -> None:
        # first try the usual suspects in environment variables, and if
        # all else fails, default to
        if value is None:
            value = os.getenv("HYDRA_BSTRAP_LOCALHOST", None)
        if value is None:
            value = os.getenv("MASTER_ADDR", None)
        if value is None:
            value = "localhost"
        # convert hostname into an IP address; does nothing if already hex
        self._main_address = gethostbyname(value)
        # for consistency, we set master address as well
        os.environ["MASTER_ADDR"] = self._main_address

    @property
    def creates_processes_externally(self) -> bool:
        """
        Override this because we rely on `mpiexec` or `mpirun` for
        the process spawning.
        """
        return True


class XPUAccelerator(Accelerator):

    """
    Implements a class for handling Intel XPU offloading, particularly the Data Center
    GPU Max Series (previously codename Ponte Vecchio).
    """

    @staticmethod
    def parse_devices(devices: int | list[int]) -> list[int]:
        """
        Parse the `trainer` input for devices and homogenize them.
        Parameters
        ----------
        devices : Union[int, List[int]]
            Single or list of device numbers to use
        Returns
        -------
        List[int]
            List of device numbers to use
        """
        if isinstance(devices, int):
            devices = [
                devices,
            ]
        return devices

    def setup_device(self, device: torch.device) -> None:
        """
        Configure the current process to use a specified device.
        Perhaps unreliably and misguiding, the IPEX implementation of this method
        tries to mirror the CUDA version but `ipex.xpu.set_device` actually refuses
        to accept anything other than an index. I've tried to work around this
        by grabbing the index from the device if possible, and just setting
        it to the first device if not using a distributed/multitile setup.
        """
        # first try and see if we can grab the index from the device
        
        index = getattr(device, "index", None)
        if index is None and not dist.is_initialized():
            index = 0
        torch.xpu.set_device(index-1)

    def teardown(self) -> None:
        # as it suggests, this is run on cleanup
        """Ensure that distributed processes close gracefully."""
        super().teardown()
        torch.xpu.empty_cache()
        if dist.is_initialized():
            dist.destroy_process_group()

    def get_device_stats(self, device) -> dict[str, any]:
        return torch.xpu.memory_stats(device)

    @staticmethod
    def get_parallel_devices(devices: list[int]) -> list[torch.device]:
        """
        Return a list of torch devices corresponding to what is available.
        Essentially maps indices to `torch.device` objects.
        Parameters
        ----------
        devices : List[int]
            List of integers corresponding to device numbers
        Returns
        -------
        List[torch.device]
            List of `torch.device` objects for each device
        """
        # import pdb; pdb.set_trace()
        # raise Exception('getting parallel devices')
        return [torch.device("xpu", i-1) for i in devices]
        # return [torch.device("xpu", i) for i in [0,1]]
        # return [torch.device("xpu", i) for i in [0]]

    @staticmethod
    def auto_device_count() -> int:
        # by default, PVC has two tiles per GPU
        return torch.xpu.device_count() 

    @staticmethod
    def is_available() -> bool:
        """
        Determines if an XPU is actually available.
        Returns
        -------
        bool
            True if devices are detected, otherwise False
        """
        try:
            return torch.xpu.device_count() != 0
        except (AttributeError, NameError):
            return False

    @classmethod
    def register_accelerators(cls, accelerator_registry) -> None:
        accelerator_registry.register(
            "xpu",
            cls,
            description="Intel Data Center GPU Max - codename Ponte Vecchio",
        )


# add PVC to the registry
AcceleratorRegistry.register("xpu", XPUAccelerator)


class SingleXPUStrategy(SingleDeviceStrategy):

    """
    This class implements the strategy for using a single XPU tile.
    """

    strategy_name = "xpu_single"

    def __init__(
        self,
        device: str | None = "xpu",
        checkpoint_io=None,
        precision_plugin=None,
    ):
        super().__init__(
            device=device,
            accelerator=XPUAccelerator(),
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
        )

    @property
    def is_distributed(self) -> bool:
        return False

    def setup(self, trainer) -> None:
        self.model_to_device()
        super().setup(trainer)

    def setup_optimizers(self, trainer) -> None:
        super().setup_optimizers(trainer)

    def model_to_device(self) -> None:
        self.model.to(self.root_device)

    @classmethod
    def register_strategies(cls, strategy_registry) -> None:
        strategy_registry.register(
            cls.strategy_name,
            cls,
            description=f"{cls.__class__.__name__}",
        )

log = logging.getLogger(__name__)

class DDPXPUStrategy(DDPStrategy):
    strategy_name = "xpu_ddp"

    def __init__(
        self,
        parallel_devices: List[torch.device] | None = None,
        cluster_environment: ClusterEnvironment | None = None,
        checkpoint_io: CheckpointIO | None = None,
        precision_plugin: MixedPrecisionPlugin | None = None,
        ddp_comm_state: object | None = None,
        ddp_comm_hook: Callable[..., Any] | None = None,
        ddp_comm_wrapper: Callable[..., Any] | None = None,
        model_averaging_period: int | None = None,
        process_group_backend: str | None = "ccl",
        timeout: timedelta | None = default_pg_timeout,
        **kwargs: Any,
    ) -> None:
        accelerator = XPUAccelerator()
        if cluster_environment is None:
            cluster_environment = IntelMPIEnvironment(main_address='127.0.0.1', main_port=random.randint(2048, 65535))
        parallel_devices = [torch.device("xpu", i) for i in range(cluster_environment.world_size())]
        super().__init__(
            accelerator,
            parallel_devices,
            cluster_environment,
            checkpoint_io,
            precision_plugin,
            ddp_comm_state,
            ddp_comm_hook,
            ddp_comm_wrapper,
            model_averaging_period,
            process_group_backend,
            timeout,
            **kwargs,
        )

    @classmethod
    def register_strategies(cls, strategy_registry) -> None:
        strategy_registry.register(
            cls.strategy_name,
            cls,
            description=f"{cls.__class__.__name__} - uses distributed data parallelism"
            " to divide data across multiple XPU tiles.",
        )

    @staticmethod
    def _init_dist_connection(
        cluster_environment: ClusterEnvironment,
        torch_distributed_backend: str,
        global_rank: Optional[int] = None,
        world_size: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Utility function to initialize distributed connection by setting env variables and initializing the
        distributed process group.

        Args:
            cluster_environment: ``ClusterEnvironment`` instance
            torch_distributed_backend: Backend to use (includes `nccl` and `gloo`)
            global_rank: Rank of the current process
            world_size: Number of processes in the group
            kwargs: Kwargs for ``init_process_group``

        Raises:
            RuntimeError:
                If ``torch.distributed`` is not available
        """
        if not torch.distributed.is_available():
            raise RuntimeError("torch.distributed is not available. Cannot initialize distributed process group")
        if torch.distributed.is_initialized():
            log.debug("torch.distributed is already initialized. Exiting early")
            return
        global_rank = global_rank if global_rank is not None else cluster_environment.global_rank()
        # world_size = world_size if world_size is not None else cluster_environment.world_size()
        local_rank = cluster_environment.local_rank()
        world_size = cluster_environment.world_size()
        # os.environ["MASTER_ADDR"] = cluster_environment.main_address
        # os.environ["MASTER_PORT"] = str(cluster_environment.main_port)

        init_method = 'tcp://%s:%s' % (os.environ["MASTER_ADDR"], cluster_environment.main_port)
        print('init_method', init_method)
        log.info(f"Initializing distributed: GLOBAL_RANK: {global_rank}, MEMBER: {global_rank + 1}/{world_size}")
        torch.distributed.init_process_group(torch_distributed_backend, rank=global_rank, world_size=world_size, init_method=init_method, **kwargs)
        # torch.distributed.init_process_group(torch_distributed_backend, rank=local_rank, world_size=world_size, init_method=init_method)
        dummy = torch.ones((5,2,), device=f"xpu:{local_rank}")
        torch.distributed.all_reduce(dummy)


    def setup_distributed(self) -> None:
        log.info(f"{self.__class__.__name__}: setting up distributed...")

        """Overrides base method so we can perform dummy all_reduce."""
        port = self.cluster_environment.main_port
        addr = self.cluster_environment.main_address
        if not dist.is_initialized():
            dist.init_process_group(
                self.process_group_backend,
                init_method=f"tcp://{addr}:{port}",
                world_size=self.cluster_environment.world_size(),
                rank=self.cluster_environment.global_rank(),
            )
        # this is to force initialization of distributed backend
        dummy = torch.ones((5, 2), device=self.root_device)
        dist.broadcast(dummy, src=0)

    @override
    def _setup_model(self, model: torch.nn.Module) -> DistributedDataParallel:
        # import pdb; pdb.set_trace()
        """Wraps the model into a :class:`~torch.nn.parallel.distributed.DistributedDataParallel` module."""
        # device_ids = self.determine_ddp_device_ids()
        device_ids = [self.cluster_environment.local_rank()]
        log.debug(f"setting up DDP model with device ids: {device_ids}, kwargs: {self._ddp_kwargs}")
        # https://pytorch.org/docs/stable/notes/cuda.html#id5
        if "cuda" in str(model.device):
            torch.cuda.current_stream()
        elif "xpu" in str(model.device):
            ctx = torch.xpu.stream(torch.xpu.Stream())
        else:
            ctx = nullcontext()
        torch.distributed.barrier(device_ids=device_ids)
        with ctx:
            ddp_model = DistributedDataParallel(module=model, device_ids=device_ids, output_device=self.cluster_environment.local_rank(), **self._ddp_kwargs)
            # ddp_model = DistributedDataParallel(module=model, device_ids=device_ids)
            return ddp_model 

    def setup(self, trainer) -> None:
        super().setup(trainer)

    def setup_optimizers(self, trainer) -> None:
        super().setup_optimizers(trainer)

    def model_to_device(self) -> None:
        self.model.to(self.root_device)
