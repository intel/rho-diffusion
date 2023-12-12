from __future__ import annotations

from typing import Any

import intel_extension_for_pytorch as ipex
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.overrides.distributed import LightningDistributedModule
from torch.nn.parallel import DistributedDataParallel


class IPEXCallback(Callback):
    def __init__(self, **ipex_kwargs: Any) -> None:
        super().__init__()
        ipex_kwargs.setdefault("level", "O1")
        ipex_kwargs.setdefault("inplace", True)
        self.ipex_kwargs = ipex_kwargs

    def on_fit_start(
        self,
        trainer,
        pl_module,
    ) -> None:
        is_ddp = isinstance(trainer.model, DistributedDataParallel)
        num_opt = len(trainer.optimizers)
        assert (
            num_opt == 1
        ), f"Only one optimizer is currently supported in IPEXCallback. Passed {num_opt}."
        opt = trainer.optimizers.pop(0)
        model, opt = ipex.optimize(
            pl_module,
            pl_module.dtype,
            opt,
            **self.ipex_kwargs,
        )
        # rewrap the model with the right constructs
        if is_ddp:
            # TODO for newer lightning (>1.9) LightningDistributedModule is removed
            model = DistributedDataParallel(LightningDistributedModule(model))
        # rewrite the trainer attributes to the wrapped things
        trainer.model = model
        trainer.optimizers = [opt]
