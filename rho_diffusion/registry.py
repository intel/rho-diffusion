from __future__ import annotations

from typing import Any

from torch import nn
from torch import optim


class Registry:
    mapping = {
        "models": {},
        "activations": {},
        "layers": {},
        "datasets": {},
        "nn": {},
        "schedules": {},
        "optimizers": {},
    }

    @classmethod
    def register_model(cls: Registry, name: str) -> callable:
        def wrapper(target: nn.Module):
            cls.mapping["models"][name] = target
            return target

        return wrapper

    @classmethod
    def register_activation(cls: Registry, name: str) -> callable:
        def wrapper(target: nn.Module):
            cls.mapping["activations"][name] = target
            return target

        return wrapper

    @classmethod
    def register_layer(cls: Registry, name: str) -> callable:
        def wrapper(target: nn.Module):
            cls.mapping["layers"][name] = target
            return target

        return wrapper

    @classmethod
    def register_dataset(cls: Registry, name: str) -> callable:
        def wrapper(target: nn.Module):
            cls.mapping["datasets"][name] = target
            return target

        return wrapper

    @classmethod
    def register_nn(cls: Registry, name: str) -> callable:
        def wrapper(target: nn.Module):
            cls.mapping["nn"][name] = target
            return target

        return wrapper

    @classmethod
    def register_schedule(cls: Registry, name: str) -> callable:
        def wrapper(target: nn.Module):
            cls.mapping["schedules"][name] = target
            return target

        return wrapper

    @classmethod
    def register_optimizer(cls: Registry, name: str) -> callable:
        def wrapper(target: nn.Module):
            cls.mapping["optimizers"][name] = target
            return target

        return wrapper

    def get(self, category: str, name: str) -> Any:
        """
        Retrieve a class/function from the registry.

        The two arguments needed specify the type of class your are looking for
        as contained in the ``Registry.categories`` property, followed by
        the name the class/function was registered as.

        Parameters
        ----------
        category : str
            Category the target class/function exists under, for example
            ``activations`` or ``datasets``.
        name : str
            Registered name of the class/function

        Returns
        -------
        Any
            Reference to the class/function

        Raises
        ------
        KeyError
            If the class/function cannot be found under the category,
            raise a ``KeyError``.
        """
        assert (
            category in self.mapping
        ), f"{category} is not a category within Registry - valid entries: {self.mapping.keys()}."
        _class = self.mapping[category].get(name, None)
        if not _class:
            raise KeyError(
                f"{name} is not a member of {category} category in Registry.",
            )
        return _class

    @property
    def categories(self) -> list[str]:
        """
        Provides a list of categories of classes included in the registry.

        Returns
        -------
        List[str]
            List of categories
        """
        return list(self.mapping.keys())

    def __repr__(self) -> str:
        """
        Formats the items contained in this registry in a quasi-presentable way.

        Returns
        -------
        str
            Formatted string representation of the registry
        """
        formatted_str = "Registry\n========"
        for key, value in self.mapping.items():
            formatted_str += f"{key}: {str(value)}\n"
        return formatted_str


registry = Registry()

# prepopulate registry with PyTorch classes

# activations
for name in [
    "ReLU",
    "SiLU",
    "Tanh",
    "Sigmoid",
    "ELU",
    "GELU",
    "PReLU",
    "Softmax",
    "LogSoftmax",
]:
    _class = getattr(nn, name)
    registry.register_activation(name)(_class)

# optimizers
for name in [
    "ASGD",
    "Adadelta",
    "Adagrad",
    "Adam",
    "AdamW",
    "Adamax",
    "LBFGS",
    "NAdam",
    "RAdam",
    "RMSprop",
    "Rprop",
    "SGD",
    "SparseAdam",
]:
    _class = getattr(optim, name)
    registry.register_optimizer(name)(_class)


# globally nn, just for redundancy
for key in dir(nn):
    # exclude nn.Module itself
    if key != "Module":
        _class = getattr(nn, key, None)
        if isinstance(_class, type) and issubclass(_class, nn.Module):
            registry.register_nn(key)(_class)
