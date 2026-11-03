"""Frozen Stage 0 data contracts from the project architecture."""

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from chaosnet._serialization import canonical_json


@dataclass(frozen=True)
class SystemSpec:
    """Variable meanings, sampling, labels, and actuator constraints.

    Field names and defaults match the architecture. ``notes`` must contain
    JSON-compatible data. As with a normal frozen dataclass, freezing prevents
    attribute reassignment; callers must also treat nested notes as read-only.
    """

    name: str
    state_names: tuple[str, ...]
    state_units: tuple[str, ...]
    param_names: tuple[str, ...]
    ext_names: tuple[str, ...]
    ctrl_names: tuple[str, ...]
    ctrl_bounds: tuple[tuple[float, float], ...]
    ctrl_rate_limit: tuple[float, ...]
    Ts: float
    window: int
    horizon_s: float
    lyap_threshold: float
    undesirable: tuple[str, ...] = ("chaotic",)
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def n_state(self) -> int:
        return len(self.state_names)

    @property
    def n_ext(self) -> int:
        return len(self.ext_names)

    @property
    def n_ctrl(self) -> int:
        return len(self.ctrl_names)

    @property
    def n_seq(self) -> int:
        """Sequence width: states + external signals + previous controls."""
        return self.n_state + self.n_ext + self.n_ctrl

    def to_dict(self) -> dict[str, Any]:
        """Return an independent, JSON-compatible representation."""
        return json.loads(canonical_json(asdict(self)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SystemSpec":
        """Restore tuple fields from a JSON/YAML dictionary.

        Unknown fields and missing required fields raise TypeError rather than
        being silently dropped. Optional fields retain their agreed defaults.
        """
        data = json.loads(canonical_json(dict(payload)))
        tuple_fields = (
            "state_names", "state_units", "param_names", "ext_names",
            "ctrl_names", "ctrl_rate_limit", "undesirable",
        )
        for key in tuple_fields:
            if key in data:
                if not isinstance(data[key], list):
                    raise TypeError(f"{key} must be a sequence")
                data[key] = tuple(data[key])
        if "ctrl_bounds" in data:
            data["ctrl_bounds"] = tuple(tuple(pair) for pair in data["ctrl_bounds"])
        return cls(**data)

    def spec_hash(self) -> str:
        """SHA-256 over canonical UTF-8 JSON, independent of Python hash seeds."""
        encoded = canonical_json(self.to_dict()).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass
class RunRecord:
    """One complete simulation run; arrays retain their original numeric dtypes.

    ``u[k]`` is the command applied during interval k. A later window builder
    must shift this channel to use past applied commands only. ``params`` and
    ``meta`` must be JSON-compatible dictionaries. No Python pickle is stored.
    """

    run_id: str
    params: dict[str, float]
    t: np.ndarray
    X: np.ndarray
    ext: np.ndarray
    u: np.ndarray
    regime: np.ndarray
    lyap: np.ndarray
    mode: str
    meta: dict[str, Any]

    def save_npz(self, path: str | Path) -> None:
        """Save all agreed fields at exactly ``path`` using compressed NPZ."""
        arrays = {
            name: np.asarray(getattr(self, name))
            for name in ("t", "X", "ext", "u", "regime", "lyap")
        }
        for name, values in arrays.items():
            if values.dtype.kind not in "biuf":
                raise TypeError(f"{name} must contain real numeric data")
        payload = {
            **arrays,
            "run_id": np.asarray(self.run_id),
            "params": np.asarray(canonical_json(self.params)),
            "mode": np.asarray(self.mode),
            "meta": np.asarray(canonical_json(self.meta)),
        }
        with Path(path).open("wb") as stream:
            np.savez_compressed(stream, **payload)

    @classmethod
    def load_npz(cls, path: str | Path) -> "RunRecord":
        """Load a RunRecord without enabling pickle or retaining an open file."""
        required = {
            "run_id", "params", "t", "X", "ext", "u", "regime", "lyap",
            "mode", "meta",
        }
        with np.load(path, allow_pickle=False) as archive:
            missing = required - set(archive.files)
            if missing:
                raise ValueError(f"RunRecord archive is missing fields: {sorted(missing)}")
            return cls(
                run_id=str(archive["run_id"].item()),
                params=json.loads(str(archive["params"].item())),
                t=archive["t"].copy(),
                X=archive["X"].copy(),
                ext=archive["ext"].copy(),
                u=archive["u"].copy(),
                regime=archive["regime"].copy(),
                lyap=archive["lyap"].copy(),
                mode=str(archive["mode"].item()),
                meta=json.loads(str(archive["meta"].item())),
            )
