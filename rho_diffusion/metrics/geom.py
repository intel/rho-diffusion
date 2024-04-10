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
