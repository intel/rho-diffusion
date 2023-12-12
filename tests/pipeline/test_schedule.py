from __future__ import annotations

import torch
from pytest import approx

from rho_diffusion.diffusion import schedule as s
from rho_diffusion.registry import registry


def test_linear_schedule():
    """Some numerical tests to make sure linear schedule is reproducible."""
    schedule = s.LinearSchedule(100, 1e-4, 0.02)
    beta_t = schedule.beta_t
    assert len(beta_t) == 100
    assert torch.is_floating_point(beta_t)
    # these should be 10x the start/end values based on scale
    assert beta_t[0] == 0.001
    assert beta_t[-1] == 0.2
    alpha_t = schedule.alpha_t
    assert torch.is_floating_point(alpha_t)
    assert alpha_t[0] == 0.999
    assert alpha_t[-1] == 0.8
    # sigma is pretty important so make sure this is correct
    sigma_t = schedule.sigma_t
    assert torch.is_floating_point(sigma_t)
    assert sigma_t[0] == 0.0
    # this one is a bit finicky so it uses approx
    assert approx(sigma_t[-1], 1e-4) == 0.4472


def test_get_from_registry():
    """Check to see if schedules are registered properly."""
    for name in s.__all__:
        _sch = registry.get("schedules", name)
        assert _sch
