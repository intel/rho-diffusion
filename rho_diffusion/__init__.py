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

__version__ = "0.1.2"

__all__ = []

from rho_diffusion import models, data, diffusion, metrics, activations

try:
    from rho_diffusion import xpu
    from rho_diffusion.xpu import SingleXPUStrategy, DDPXPUStrategy
    from lightning.pytorch.strategies import StrategyRegistry

    StrategyRegistry.register("xpu_single", SingleXPUStrategy)
    StrategyRegistry.register("xpu_ddp", DDPXPUStrategy)
except (ModuleNotFoundError, ImportError) as error:
    print(error)

