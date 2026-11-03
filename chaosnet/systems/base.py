"""Abstract adapter contract shared by simulation and control contributors."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from chaosnet.data.schema import SystemSpec


class SystemAdapter(ABC):
    """Describe a system without coupling its equations to an integrator.

    State, external-input and control arrays use their last axis for channels;
    leading axes may represent a simulation batch. Parameter values may be
    scalars or arrays broadcastable across that batch.
    """

    @abstractmethod
    def spec(self, cfg: Mapping[str, Any]) -> SystemSpec:
        """Return the variable names, units, dimensions and constraints for cfg."""

    @abstractmethod
    def rhs(
        self, t: float, x: np.ndarray, p: Mapping[str, Any],
        e: np.ndarray, u: np.ndarray,
    ) -> np.ndarray:
        """Return state derivatives with the same shape as x, without stepping time."""

    @abstractmethod
    def ext_signals(
        self, t: float | np.ndarray, params: Mapping[str, Any],
        cfg: Mapping[str, Any],
    ) -> np.ndarray:
        """Return external signals in spec.ext_names order at the supplied times."""

    @abstractmethod
    def candidate_controllers(self) -> dict[str, Callable[..., Callable[..., np.ndarray]]]:
        """Return named expert-controller factories; callers own controller state."""
