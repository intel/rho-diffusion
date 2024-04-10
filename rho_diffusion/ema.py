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

import copy
import math
from collections import OrderedDict

import torch
from torch import nn


class ExponentialMovingAverage(nn.Module):
    def __init__(self, model, decay=0.9999) -> None:
        super().__init__()
        self.model = model
        self.ema_model = copy.deepcopy(model).eval()
        self.decay_func = lambda x: decay * (1 - math.exp(-x / 2000))
        self.step_id = 0
        self.current_ema_frac = 0.0

        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    def update(self):
        with torch.no_grad():
            self.step_id += 1
            current_frac = self.decay_func(self.step_id)
            self.current_ema_frac = current_frac

            model_params = OrderedDict(self.model.named_parameters())
            shadow_params = OrderedDict(self.ema_model.named_parameters())

            # check if both model contains the same set of keys
            assert model_params.keys() == shadow_params.keys()

            for name, param in model_params.items():
                # see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
                # shadow_variable -= (1 - decay) * (shadow_variable - variable)
                shadow_params[name].sub_(
                    (1.0 - current_frac) * (shadow_params[name] - param),
                )

            model_buffers = OrderedDict(self.model.named_buffers())
            shadow_buffers = OrderedDict(self.ema_model.named_buffers())

            # check if both model contains the same set of keys
            assert model_buffers.keys() == shadow_buffers.keys()

            for name, buffer in model_buffers.items():
                # buffers are copied
                shadow_buffers[name].copy_(buffer)

            # for i, item in self.ema_model.state_dict().items():
            #     if item.dtype.is_floating_point:
            #         print('copying...')
            #         item = item * current_frac + (1 - current_frac) * self.model.state_dict()[i].detach()

    def forward(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)

    def denoise_process(self, *args, **kwargs):
        return self.ema_model.denoise_process(*args, **kwargs)
