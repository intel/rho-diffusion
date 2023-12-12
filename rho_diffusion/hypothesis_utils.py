"""
Chunks of code are obtained from this repository:

https://github.com/janfreyberg/torch-hypothesis/blob/master/src/torch_hypothesis/__init__.py
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from hypothesis import strategies as st
from hypothesis.extra import numpy as npst


def torch_dtype_to_numpy_dtype(dtype):
    return np.dtype(str(dtype).replace("torch.", ""))


def is_numeric(value: Any) -> bool:
    """
    Test if an object is numeric or not.

    Parameters
    ----------
    value : Any
        Target to test

    Returns
    -------
    bool
        True if the object can be converted to a float,
        otherwise False.
    """
    try:
        float(value)
        return True
    except ValueError:
        return False


def from_range_value_or_choice(parameter, draw):
    # if strategy, just draw:
    if isinstance(parameter, st.SearchStrategy):
        return draw(parameter)
    # if len 2 sequence of numbers, draw from strat:
    if (
        isinstance(parameter, Sequence)
        and len(parameter) == 2
        and all(is_numeric(param) or param is None for param in parameter)
    ):
        if all(isinstance(param, int) or param is None for param in parameter):
            return draw(st.integers(min_value=parameter[0], max_value=parameter[1]))
        elif all(isinstance(param, float) or param is None for param in parameter):
            return draw(st.floats(min_value=parameter[0], max_value=parameter[1]))
    # if some other sequence, return one element from it:
    elif isinstance(parameter, Sequence):
        return draw(st.sampled_from(parameter))
    else:
        return parameter


@st.composite
def torch_tensor(draw, shape, values, dtype):
    shape = (from_range_value_or_choice(size, draw) for size in shape)
    dtype = from_range_value_or_choice(dtype, draw)
    np_array = draw(
        npst.arrays(torch_dtype_to_numpy_dtype(dtype), shape, elements=values),
    )
    return torch.as_tensor(np_array)
