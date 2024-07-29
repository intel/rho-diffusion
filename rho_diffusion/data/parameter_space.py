import torch
from typing import Any, Union
from collections import OrderedDict
import itertools
from abc import ABC
import numpy as np 


def sample_from_discrete_parameter_space(param_dict: dict, batch_size: int, random=True, device=None) -> torch.Tensor:
    keys, values = zip(*param_dict.items())
    combinations = torch.tensor([v for v in itertools.product(*values)], device=device)
    if random:
        idx = torch.randint(low=0, high=combinations.shape[0], size=(batch_size,), device=device)
    else:
        idx = torch.arange(start=0, end=batch_size, step=1, device=device)
    return combinations[idx]


class AbstractParameterSpace(ABC):
    

    def __init__(self, param_dict=None, sampler=None):
        if param_dict is not None:
            self.param_dict = param_dict
        else:
            self.param_dict = OrderedDict()

        self.sampler = sampler

    def set(self, param_dict: Union[dict, OrderedDict]) -> None:
        self.param_dict = param_dict

    @property
    def parameters(self) -> list:
        return self.param_dict.keys()

    def sample(self, num_samples, device=None):
        raise NotImplementedError('Method sample() is not implemented.')

    def size(self):
        raise NotImplementedError('Method size() is not implemented.')

    def push_parameter(self, key: str, value: Any) -> None:
        raise NotImplementedError('Method push_parameter() is not implemented.')

    def __repr__(self) -> str:
        return self.param_dict.__repr__()

    def __getitem__(self, key) -> Any:
        return self.param_dict[key]

    def __setitem__(self, key, value):
        self.param_dict[key] = value 

    def __len__(self) -> int:
        return len(self.param_dict)

    def items(self):
        return self.param_dict.items()
    
    def values(self):
        return self.param_dict.values()
    
    def keys(self):
        return self.param_dict.keys()


class DiscreteParameterSpace(AbstractParameterSpace):

    def __init__(self, param_dict=None, sampler=None):
        super().__init__(param_dict=param_dict, sampler=sampler)
        if sampler is None:
            self.sampler = sample_from_discrete_parameter_space
        else:
            self.sampler = sampler

    def sample(self, num_samples, device=None):
        return self.sampler(self.param_dict, batch_size=num_samples, random=True, device=device)

    def size(self):
        keys, values = zip(*self.param_dict.items())
        return len([v for v in itertools.product(*values)])
    
    def push_parameter(self, key: str, value: Any) -> None:
        if self.param_dict[key] is None:
            self.param_dict[key] = []
        elif isinstance(Union[list, np.ndarray]):
            for i, v in enumerate(value):
                if v not in self.param_dict[key]:
                    self.param_dict[key].append(v)
        elif value not in self.param_dict[key]:
            self.param_dict[key].append(value)

        


