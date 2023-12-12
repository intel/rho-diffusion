from __future__ import annotations

import pytest
import torch

from rho_diffusion.models.unet_v2 import UNet
from rho_diffusion.registry import registry


def test_unet_in_registry() -> type[UNet]:
    model_class = registry.get("models", "UNetv2")
    assert model_class, "UNetv2 was not found in registry!"


@pytest.mark.dependency()
@pytest.fixture()
def test_unet_init() -> UNet:
    unet = UNet(
        data_shape=16,
        in_channels=3,
        model_channels=32,
        out_channels=3,
        num_res_blocks=2,
    )
    return unet


@pytest.mark.dependency(depends=["test_unet_init"])
def test_unet_forward(test_unet_init):
    model = test_unet_init
    x = torch.rand(8, 3, 24, 16)
    timesteps = torch.arange(8)
    with torch.inference_mode():
        pred_y = model(x, timesteps)
        assert isinstance(pred_y, torch.Tensor)
        # this test is kind of confusing, but just making sure there
        # are no nans in the predictions
        assert ~torch.isnan(pred_y).all()
