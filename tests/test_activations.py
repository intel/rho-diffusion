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
