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


import torch 
from torch import nn 
from torch import functional as F 
import numpy as np 
from collections import OrderedDict
from typing import Union
import itertools

from rho_diffusion.registry import registry
from rho_diffusion.data.parameter_space import AbstractParameterSpace


@registry.register_layer("MultiEmbeddings")
class MultiEmbeddings(nn.Module):
    """Generates embeddings from an N-dimensional parameter space defined in a dictionary structure.
    """
    def __init__(self, 
                 parameter_space: AbstractParameterSpace = None, 
                 embedding_dim: int = 512,
                 parameter_space_dim: int = 3,
                 embedding_size: Union[int, list, dict, OrderedDict] = None) -> None:
        
        """Initialize a `MultiEmbeddings` object.

        Args:
            parameter_space (Union[dict, OrderedDict], optional): The parameter space dict. If set, the `parameter_space_dim` and `embedding_size` will be inferred from it. Defaults to None.
            embedding_dim (int, optional): The length of each embedding vector. Defaults to 512.
            parameter_space_dim (int, optional): The dimensionality of the parameter space. Defaults to 3.
            embedding_size (Union[int, list], optional): The total number of embeddings in each parameter space dimension. Defaults to None.
        """
        super().__init__()

        self.embedding_layers = nn.ModuleDict()
        self.label_encoders = OrderedDict()
        self.parameter_space = parameter_space
        self.embedding_dim = embedding_dim
        self.__label_encoder_fitted = False
        if parameter_space is not None and len(parameter_space) > 0:
            # Initialize the embeddings using the parameter space dict
            for key, value in self.parameter_space.items():
                self.__label_encoder_fitted = True
                self.embedding_layers[key] = nn.Embedding(num_embeddings=len(value), embedding_dim=self.embedding_dim)
        elif embedding_size is not None:
            # if the parameter space is not yet known, the MultiEmbeddings object can also be setup if `parameter_space_dim` and/or `embedding_size` are known
            if isinstance(embedding_size, int):
                assert parameter_space_dim is not None, 'parameter_space_dim should be an int input when embedding_size is an int'
                for i in parameter_space_dim:
                    self.embedding_layers[i] = nn.Embedding(num_embeddings=embedding_size, embedding_dim=embedding_dim)
            elif isinstance(embedding_size, list):
                for i in len(embedding_size):
                    self.embedding_layers[i] = nn.Embedding(num_embeddings=embedding_size[i], embedding_dim=embedding_dim)
            elif isinstance(embedding_size, dict) or isinstance(embedding_dim, OrderedDict):
                for key, value in embedding_size.items():
                    self.embedding_layers[key] = nn.Embedding(num_embeddings=value, embedding_dim=embedding_dim)
                

    # def transform_to_categorical(self, params: dict) -> torch.Tensor:
    #     """Transforms a batch of parameters to a 2D categorical tensor.

    #     Args:
    #         params (dict): A dict where different parameter vectors are indexed by the correspond key

    #     Returns:
    #         torch.Tensor: A 2D tensor where each row stores the categorical encoded values of the parameter
    #     """
    #     # batch_size = 0
    #     categorical_values = OrderedDict()
    #     keys, values = zip(*params.items())
    #     combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    #     # for i in range(len(combinations)):
    #     for key, param_vec in params.items():
    #         if self.__label_encoder_fitted:
    #             categorical_values[key] = np.array(self.label_encoders[key].transform(param_vec))
    #         else:
    #             print(self.label_encoders)
    #             categorical_values[key] = np.array(self.label_encoders[key].fit_transform(param_vec))
    #         # if batch_size == 0:
    #         #     batch_size = len(param_vec)
    #         # else:
    #         #     # make sure that each feature in the parameter space has the same length with other features
    #         #     assert batch_size == len(param_vec)
        
    #     categorical_tensor = torch.zeros((batch_size, len(categorical_values)), dtype=torch.int)

    #     i = 0
    #     for key, cat_vec in categorical_values.items():
    #         categorical_tensor[:, i] = torch.LongTensor(cat_vec)
    #         i += 1

    #     return categorical_tensor


    


    def forward(self, y: torch.Tensor) -> torch.Tensor:
        assert y.dim() == 2
        emb = None
        i = 0
        for key, layer in self.embedding_layers.items():
            # Use the index of each element of the i-th feature in the parameter space as categorical
            # print(y[:, i], self.parameter_space[key])

            categorical = torch.where(y[:, i][:, None] == torch.tensor(self.parameter_space[key], device=y.device)[None, :])[1]
            if emb is None:
                emb = layer(categorical)
            else:
                # print(categorical.shape, emb.shape)
                emb += layer(categorical)
            i += 1
        return emb 


class ClassifierGuidance(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = None 
        self.classifier_scale = 1.0

    def forward(self, x, t, y=None):
        assert y is not None
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = self.classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return torch.autograd.grad(selected.sum(), x_in)[0] * self.classifier_scale