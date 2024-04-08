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

from rho_diffusion.registry import registry

test_space = registry.mapping["activations"]


@pytest.mark.parametrize("classname", test_space)
def test_activation_functions(classname):
    x = torch.rand(8, 10)
    class_type = registry.get("activations", classname)
    activation = class_type()
    with torch.inference_mode():
        activation(x)
