from __future__ import annotations

import os
from socket import gethostbyname
from typing import List, Any, Callable
from datetime import timedelta
import intel_extension_for_pytorch as ipex
import torch
from lightning.pytorch.accelerators import Accelerator
from lightning.pytorch.accelerators import AcceleratorRegistry
from lightning.pytorch.plugins.environments import LightningEnvironment
from lightning.pytorch.plugins.precision import NativeMixedPrecisionPlugin as MixedPrecisionPlugin
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.strategies import SingleDeviceStrategy
from lightning_lite.plugins import CheckpointIO
from lightning_lite.plugins import ClusterEnvironment
from lightning_lite.plugins.collectives.torch_collective import default_pg_timeout
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_only, rank_zero_warn
from lightning.lite.utilities.distributed import (
    _distributed_available,
    _get_default_process_group_backend_for_device,
    _init_dist_connection,
    _sync_ddp_if_available,
)
from lightning.lite.utilities.seed import reset_seed
from torch import distributed as dist

import logging 


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
        return int(os.environ["PMI_SIZE"])

    def local_rank(self) -> int:
        return int(os.environ["MPI_LOCALRANKID"])

    def global_rank(self) -> int:
        return int(os.environ["PMI_RANK"])

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
        torch.xpu.set_device(index)

    def teardown(self) -> None:
        # as it suggests, this is run on cleanup
        torch.xpu.empty_cache()

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
        return [torch.device("xpu", i) for i in devices]

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
            cluster_environment = IntelMPIEnvironment()
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

    def setup_distributed(self) -> None:
        log.detail(f"{self.__class__.__name__}: setting up distributed...")
        reset_seed()
        self.set_world_ranks()
        rank_zero_only.rank = self.global_rank
        self._process_group_backend = self._get_process_group_backend()
        assert self.cluster_environment is not None
        _init_dist_connection(self.cluster_environment, 
                              self._process_group_backend, 
                              timeout=self._timeout, 
                              init_method='tcp://127.0.0.1:24566')


class XPUBF16Plugin(MixedPrecisionPlugin):
    def __init__(self):
        super().__init__(torch.bfloat16, "xpu")

    def auto_cast_context_manager(self):
        """
        Overrides the default behavior, which relies on `torch.amp` where only
        CPU and CUDA backends are supported. This uses the `xpu.amp` interface
        explicitly, as done in the IPEX documentation.
        """
        return torch.xpu.amp.autocast(self.device, enabled=True, dtype=torch.bfloat16)
