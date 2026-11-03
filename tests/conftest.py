"""Shared examples of the frozen contracts."""

import pytest

from chaosnet.data.schema import SystemSpec


@pytest.fixture
def spec() -> SystemSpec:
    return SystemSpec(
        name="contract_test_v1",
        state_names=("x", "v"),
        state_units=("m", "m/s"),
        param_names=("c", "k", "beta", "omega"),
        ext_names=("F", "cos_wt", "sin_wt"),
        ctrl_names=("u_force",),
        ctrl_bounds=((-0.5, 0.5),),
        ctrl_rate_limit=(2.0,),
        Ts=0.1,
        window=64,
        horizon_s=2.0,
        lyap_threshold=0.01,
        notes={"source": "test", "settings": {"second": 2, "first": 1}},
    )
