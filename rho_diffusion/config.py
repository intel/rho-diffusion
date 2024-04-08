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

import json
from pathlib import Path
from typing import Any
from typing import Union

from pydantic import BaseModel
from pydantic import validator

from rho_diffusion.utils import number_cast_dict

# define additional types for convenience
GenericParameter = Union[str, float]
IterableParameter = list[GenericParameter]


class ComponentConfig(BaseModel):
    """
    Defines a configuration model for specific components, which
    are essentially just name/kwarg mappings.
    """

    name: str
    kwargs: dict[str, Union[GenericParameter, IterableParameter]]

    @validator("kwargs")
    def number_cast(cls, kwargs):
        kwargs = number_cast_dict(kwargs)
        return kwargs


class TrainingConfig(BaseModel):
    """
    Defines a configuration model for training set up.
    """

    device: str
    loss_fn: str = "MSELoss"
    ema_decay: float = 0.0
    batch_size: int = 32
    seed: int = 777
    min_epochs: int = 1
    max_epochs: int = 999
    save_checkpoint_every_n_epochs: int = 10
    sample_every_n_epochs: int = 5


class InferenceConfig(BaseModel):
    """
    Defines a configuration model for inference set up.
    """

    device: str
    checkpoint: Union[str, Path]
    parameter_space: dict[str, Any]
    cache_file: Union[str, Path, None] = None
    plot_output_file: Union[str, Path, None] = None
    seed: int = 777


class ExperimentConfig(BaseModel):
    """
    Defines a configuration model for a collective experiment set up.

    This basically ensures everything is well defined, and value types
    are properly scoped and validated before mapping onto scripts.
    """

    experiment: str
    model: ComponentConfig
    dataset: ComponentConfig
    optimizer: ComponentConfig
    lr_scheduler: ComponentConfig
    noise_schedule: ComponentConfig
    training: TrainingConfig
    inference: InferenceConfig

    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> ExperimentConfig:
        """
        Convenience constructor method to set up an experiment config
        from a JSON file on disk.
        """
        if isinstance(json_path, str):
            json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"Specified config file not found: {json_path}")
        with open(json_path) as read_file:
            configuration = json.load(read_file)
        configuration = ExperimentConfig(**configuration)
        return configuration
