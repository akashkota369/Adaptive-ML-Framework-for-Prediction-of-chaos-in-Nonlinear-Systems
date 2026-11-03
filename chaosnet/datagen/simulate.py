"""Batched NumPy RK4, with continuously evaluated forcing and held controls."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from chaosnet.data.schema import RunRecord, SystemSpec
from chaosnet.data.validate import validate
from chaosnet.systems.base import SystemAdapter


def parameter_batch(p_batch, batch_size: int, names: tuple) -> dict[str, np.ndarray]:
    """Normalize a parameter mapping or list of per-run mappings to batch vectors."""
    if isinstance(p_batch, Mapping):
        if set(p_batch) != set(names):
            raise ValueError("Physical parameter names do not match the specification")
        result = {name: np.broadcast_to(np.asarray(p_batch[name], dtype=float), (batch_size,)).copy() for name in names}
    else:
        if len(p_batch) != batch_size or any(set(p) != set(names) for p in p_batch):
            raise ValueError("One complete physical parameter mapping is required per run")
        result = {name: np.array([p[name] for p in p_batch], dtype=float) for name in names}
    if any(not np.all(np.isfinite(values)) for values in result.values()):
        raise ValueError("Physical parameters must be finite")
    return result


def rk4_step(adapter, t, x, p, e_fn, u, dt) -> np.ndarray:
    """One vectorized RK4 step; t and dt can be per-trajectory vectors."""
    step = np.asarray(dt)[..., None]
    k1 = adapter.rhs(t, x, p, e_fn(t), u)
    k2 = adapter.rhs(t + dt / 2, x + step * k1 / 2, p, e_fn(t + dt / 2), u)
    k3 = adapter.rhs(t + dt / 2, x + step * k2 / 2, p, e_fn(t + dt / 2), u)
    k4 = adapter.rhs(t + dt, x + step * k3, p, e_fn(t + dt), u)
    return x + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def advance(adapter, t, x, p, e_fn, u, interval, substeps: int = 5) -> np.ndarray:
    """Integrate an interval with zero-order-hold u and fixed internal substeps."""
    if not isinstance(substeps, int) or substeps < 1:
        raise ValueError("substeps must be a positive integer")
    dt = np.asarray(interval) / substeps
    for j in range(substeps):
        x = rk4_step(adapter, t + j * dt, x, p, e_fn, u, dt)
    return x


def simulate_batch(
    adapter: SystemAdapter, p_batch, x0, e_fn: Callable,
    controller: Callable | None, n_steps: int, *, spec: SystemSpec,
    substeps: int = 5, seed: int = 0, t0: float = 0.0,
    run_ids: Sequence[str] | None = None, metadata: Sequence[Mapping[str, Any]] | None = None,
) -> list[RunRecord]:
    """Simulate n_steps intervals, returning T=n_steps+1 measured samples.

    e_fn(t) returns (batch, n_ext), or a broadcastable channel vector.
    controller(t, x_batch, p_batch) returns physical-unit (batch, n_ctrl)
    commands; None means zero force. Supplied commands must already obey
    bounds and slew limits: this numerical layer never changes the command.

    Raw records have label_status='unlabeled' and an all-false label_valid mask.
    Their finite zero label storage is unavailable data, never a chaos claim.
    label_runs must run before a dataset can be saved. The final u sample
    repeats the last applied command; no extra interval is integrated.
    """
    validate([], spec)
    if not isinstance(n_steps, int) or n_steps < 1 or not np.isfinite(t0):
        raise ValueError("n_steps must be positive and t0 finite")
    x = np.asarray(x0, dtype=float).copy()
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or x.shape[1] != spec.n_state or len(x) == 0 or not np.all(np.isfinite(x)):
        raise ValueError("x0 must contain finite states with shape (batch, n_state)")
    batch = len(x)
    p = parameter_batch(p_batch, batch, spec.param_names)
    if "omega" in p and np.any(p["omega"] <= 0):
        raise ValueError("Forcing frequencies must be positive")
    ids = list(run_ids) if run_ids is not None else [f"simulation-{seed}-{i:04d}" for i in range(batch)]
    meta = list(metadata) if metadata is not None else [{} for _ in range(batch)]
    if len(ids) != batch or len(set(ids)) != batch or len(meta) != batch:
        raise ValueError("Unique run IDs and metadata must match the batch size")
    times = t0 + np.arange(n_steps + 1) * spec.Ts
    states = np.empty((batch, n_steps + 1, spec.n_state))
    external = np.empty((batch, n_steps + 1, spec.n_ext))
    controls = np.empty((batch, n_steps + 1, spec.n_ctrl))
    bounds = np.asarray(spec.ctrl_bounds)
    previous = None
    for k, time in enumerate(times):
        states[:, k] = x
        e = np.broadcast_to(np.asarray(e_fn(float(time)), dtype=float), (batch, spec.n_ext))
        if not np.all(np.isfinite(e)):
            raise ValueError("External signals must be finite")
        external[:, k] = e
        if k == n_steps:
            controls[:, k] = previous
            break
        command = np.zeros((batch, spec.n_ctrl)) if controller is None else np.asarray(controller(float(time), x.copy(), p), dtype=float)
        command = np.broadcast_to(command, (batch, spec.n_ctrl)).copy()
        if not np.all(np.isfinite(command)) or np.any(command < bounds[:, 0]) or np.any(command > bounds[:, 1]):
            raise ValueError("Control command is non-finite or outside actuator bounds")
        if previous is not None and np.any(np.abs(command - previous) > np.asarray(spec.ctrl_rate_limit) * spec.Ts + 1e-12):
            raise ValueError("Control command exceeds the actuator rate limit")
        controls[:, k] = command
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            try:
                x = advance(adapter, float(time), x, p, e_fn, command, spec.Ts, substeps)
            except FloatingPointError as exc:
                raise ValueError(f"Simulation diverged at sample {k}") from exc
        if not np.all(np.isfinite(x)) or np.any(np.abs(x) > 1e6):
            raise ValueError(f"Simulation diverged at sample {k}")
        previous = command
    return [RunRecord(
        run_id=ids[i], params={name: float(values[i]) for name, values in p.items()},
        t=times.copy(), X=states[i].copy(), ext=external[i].copy(), u=controls[i].copy(),
        regime=np.zeros(n_steps + 1, dtype=np.int8), lyap=np.zeros(n_steps + 1),
        mode="open_loop" if controller is None else "candidate_trial",
        meta={**dict(meta[i]), "synthetic": False, "seed": int(seed), "spec_hash": spec.spec_hash(),
              "integrator": {"name": "rk4", "substeps": substeps, "Ts": spec.Ts},
              "simulation": {"n_steps": n_steps, "duration_s": float(n_steps * spec.Ts), "t0": float(t0)},
              "initial_state": states[i, 0].tolist(), "label_status": "unlabeled",
              "label_valid": [False] * (n_steps + 1)},
    ) for i in range(batch)]
