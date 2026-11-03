"""The adapter is an abstract contract, not an accidental runnable system."""

import inspect

import pytest

from chaosnet.systems.base import SystemAdapter


def test_adapter_requires_all_agreed_methods():
    assert SystemAdapter.__abstractmethods__ == {
        "spec", "rhs", "ext_signals", "candidate_controllers",
    }
    assert list(inspect.signature(SystemAdapter.rhs).parameters) == [
        "self", "t", "x", "p", "e", "u",
    ]
    assert list(inspect.signature(SystemAdapter.ext_signals).parameters) == [
        "self", "t", "params", "cfg",
    ]
    with pytest.raises(TypeError, match="abstract"):
        SystemAdapter()
