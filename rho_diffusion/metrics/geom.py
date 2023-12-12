from __future__ import annotations

from geomloss import SamplesLoss
from torch import nn
from torch import Tensor

from rho_diffusion.registry import registry


@registry.register_layer("WassersteinWrapper")
class WassersteinWrapper(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.metric = SamplesLoss("sinkhorn", p=1, blur=0.01)

    def forward(self, pred_data: Tensor, true_data: Tensor) -> Tensor:
        assert pred_data.shape == true_data.shape
        # flatten the data into 1D
        return self.metric(pred_data.flatten(1), true_data.flatten(1))
