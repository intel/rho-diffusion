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

import pytest
import torch
from lightning import pytorch as pl
from torch import nn

from rho_diffusion import models
from rho_diffusion.diffusion import DDPM
from rho_diffusion.registry import registry

pl.seed_everything(21516)


@pytest.fixture(scope="session")
def base_ddpm():
    """fixture for testing if a DDPM class can be instantiated"""
    unet_class = registry.get("models", "UNetv2")
    kwargs = {
        "image_size": 16,
        "in_channels": 3,
        "model_channels": 32,
        "out_channels": 3,
        "num_res_blocks": 2,
    }
    schedule_class = registry.get("schedules", "LinearSchedule")
    schedule = schedule_class(100, 1e-4, 0.02)
    ddpm = DDPM(unet_class, kwargs, schedule, nn.MSELoss)
    return ddpm


def test_ddpm_forward(base_ddpm):
    """test the forward process of a DDPM"""
    mean, var = base_ddpm.forward_process(torch.rand(1, 3, 16, 16))
    assert torch.isfinite(mean).all()
    assert torch.isfinite(var).all()


def test_ddpm_train_step(base_ddpm):
    """test the noise prediction of a DDPM"""
    test_batch = {"data": torch.rand(1, 3, 16, 16), "labels": torch.LongTensor([1])}
    loss = base_ddpm.training_step(test_batch)
    assert loss
