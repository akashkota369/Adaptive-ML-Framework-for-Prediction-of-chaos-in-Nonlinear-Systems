"""Forced Duffing oscillator: x'' + c*x' + k*x + beta*x**3 = F*cos(omega*t) + u."""

from collections.abc import Mapping
from typing import Any

import numpy as np

from chaosnet.data.schema import SystemSpec
from chaosnet.systems.base import SystemAdapter


class DuffingAdapter(SystemAdapter):
    """Mass-normalized Duffing equations with one additive force actuator."""

    def spec(self, cfg: Mapping[str, Any]) -> SystemSpec:
        parameters = cfg.get("parameters", {})
        omega = float(parameters.get("omega", 1.2))
        if not np.isfinite(omega) or omega <= 0:
            raise ValueError("omega must be positive and finite")
        sampling = cfg.get("sampling", {})
        control = cfg.get("control", {})
        period = 2 * np.pi / omega
        samples = sampling.get("samples_per_period", 32)
        if not isinstance(samples, int) or samples < 8:
            raise ValueError("samples_per_period must be an integer >= 8")
        bounds = tuple(float(v) for v in control.get("bounds", [-0.5, 0.5]))
        rate = float(control.get("rate_limit", 2.0))
        if len(bounds) != 2 or not np.all(np.isfinite(bounds)) or not bounds[0] <= 0 <= bounds[1] or bounds[0] == bounds[1]:
            raise ValueError("Duffing control bounds must be finite, increasing, and include zero")
        if not np.isfinite(rate) or rate < 0:
            raise ValueError("Control rate_limit must be non-negative and finite")
        return SystemSpec(
            name="duffing_v1", state_names=("x", "v"), state_units=("m", "m/s"),
            param_names=("c", "k", "beta", "omega"),
            ext_names=("F", "cos_wt", "sin_wt"), ctrl_names=("u_force",),
            ctrl_bounds=(bounds,), ctrl_rate_limit=(rate,),
            Ts=float(sampling.get("Ts", period / samples)),
            window=sampling.get("window", 64),
            horizon_s=float(sampling.get("horizon_s", sampling.get("horizon_periods", 20) * period)),
            lyap_threshold=float(cfg.get("labeling", {}).get("threshold", 0.01)),
            notes={"mass_normalized": True, "nominal_forcing_period": float(period),
                   "control_units": "force per unit mass", "parameter_order": ["c", "k", "beta", "omega"]},
        )

    def rhs(self, t, x, p, e, u) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        e = np.asarray(e, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)
        if x.shape[-1:] != (2,) or e.shape[-1:] != (3,) or u.shape[-1:] != (1,):
            raise ValueError("Duffing requires 2 state, 3 external, and 1 control channels")
        position, velocity = x[..., 0], x[..., 1]
        acceleration = (
            -np.asarray(p["c"]) * velocity - np.asarray(p["k"]) * position
            - np.asarray(p["beta"]) * position**3 + e[..., 0] * e[..., 1] + u[..., 0]
        )
        return np.stack((np.broadcast_to(velocity, acceleration.shape), acceleration), axis=-1)

    def ext_signals(self, t, params, cfg) -> np.ndarray:
        """Evaluate forcing continuously, including RK4 intermediate times.

        A ramp changes only amplitude F; c, k, beta, and omega stay static.
        Values before/after the ramp hold its endpoint amplitudes.
        """
        forcing = cfg.get("forcing", {"kind": "constant", "amplitude": 0.3})
        times, omega = np.broadcast_arrays(np.asarray(t, dtype=float), np.asarray(params["omega"], dtype=float))
        kind = forcing.get("kind", "constant")
        if kind == "constant":
            amplitude = np.asarray(forcing["amplitude"], dtype=float)
        elif kind == "ramp":
            start, stop = float(forcing["start_time"]), float(forcing["end_time"])
            if not np.isfinite(start + stop) or stop <= start:
                raise ValueError("Ramp end_time must be greater than start_time")
            fraction = np.clip((times - start) / (stop - start), 0.0, 1.0)
            amplitude = np.asarray(forcing["start_amplitude"]) + fraction * (
                np.asarray(forcing["end_amplitude"]) - np.asarray(forcing["start_amplitude"])
            )
        else:
            raise ValueError(f"Unknown forcing kind: {kind}")
        amplitude, phase = np.broadcast_arrays(amplitude, omega * times)
        return np.stack((amplitude, np.cos(phase), np.sin(phase)), axis=-1)

    def candidate_controllers(self) -> dict:
        """No expert-controller families are registered in Stage 1.

        The simulator accepts supplied physical-unit commands. Expert search
        and the feedback-control framework belong to their later stage.
        """
        return {}
