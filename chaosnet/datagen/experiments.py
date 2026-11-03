"""Seeded Duffing sweeps and slowly ramped forcing experiments."""

from copy import deepcopy
from dataclasses import fields
import hashlib
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from chaosnet._serialization import canonical_json
from chaosnet.configs.loader import config_hash
from chaosnet.data.validate import validate_dataset
from chaosnet.datagen.labels import LyapunovSettings, label_runs
from chaosnet.datagen.simulate import simulate_batch
from chaosnet.systems.duffing import DuffingAdapter


def source_hash() -> str:
    """Digest the numerical/data source used to produce an experiment."""
    package = Path(__file__).resolve().parents[1]
    relative_paths = ["systems/duffing.py", "datagen/simulate.py", "datagen/labels.py",
                      "datagen/experiments.py", "datagen/dataset.py", "data/schema.py", "data/validate.py"]
    digest = hashlib.sha256()
    for name in sorted(relative_paths):
        digest.update(name.encode())
        digest.update((package / name).read_bytes())
    return digest.hexdigest()


def make_duffing_runs(cfg: dict):
    """Return (SystemSpec, labeled runs) without writing files or altering cfg."""
    cfg = deepcopy(cfg)
    if cfg.get("system") != "duffing":
        raise ValueError("This experiment requires system: duffing")
    seed = cfg.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    adapter = DuffingAdapter()
    spec = adapter.spec(cfg)
    exp = cfg["experiments"]
    count = exp["n_runs"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("n_runs must be a positive integer")
    nominal_period = 2 * np.pi / cfg["parameters"]["omega"]
    duration = float(exp.get("duration_s", exp.get("duration_periods", 80) * nominal_period))
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be positive and finite")
    n_steps = int(np.ceil(duration / spec.Ts - 1e-12))
    if n_steps + 1 < spec.window + int(np.ceil(spec.horizon_s / spec.Ts)):
        raise ValueError("Run duration is too short for the configured window and prediction horizon")
    actual_duration = n_steps * spec.Ts
    fraction = exp.get("ramp_fraction", 0.5)
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("ramp_fraction must be between zero and one")
    n_ramps = int(np.floor(count * fraction))
    n_steady = count - n_ramps
    grid = np.asarray(exp["steady_amplitudes"], dtype=float)
    ends = np.asarray(exp["ramp_amplitudes"], dtype=float)
    if grid.ndim != 1 or len(grid) == 0 or not np.all(np.isfinite(grid)) or ends.shape != (2,) or not np.all(np.isfinite(ends)):
        raise ValueError("Forcing amplitudes must be finite; ramps need two endpoints")
    start_fraction, end_fraction = exp.get("ramp_start_fraction", 0.1), exp.get("ramp_end_fraction", 0.9)
    if not 0 <= start_fraction < end_fraction <= 1:
        raise ValueError("Ramp fractions must satisfy 0 <= start < end <= 1")
    x0 = np.empty((count, 2))
    for axis, key in enumerate(("initial_position", "initial_velocity")):
        limits = np.asarray(exp[key], dtype=float)
        if limits.shape != (2,) or not np.all(np.isfinite(limits)) or limits[0] > limits[1]:
            raise ValueError(f"Invalid {key} interval")
        x0[:, axis] = rng.uniform(*limits, size=count)
    parameters = [dict(cfg["parameters"]) for _ in range(count)]
    ranges = exp.get("parameter_ranges", {})
    if not isinstance(ranges, dict) or not set(ranges) <= set(spec.param_names):
        raise ValueError("parameter_ranges must name static physical parameters")
    if ranges:
        names = sorted(ranges)
        unit = qmc.LatinHypercube(d=len(names), seed=seed).random(count)
        for column, name in enumerate(names):
            limits = np.asarray(ranges[name], dtype=float)
            if limits.shape != (2,) or not np.all(np.isfinite(limits)) or limits[0] > limits[1]:
                raise ValueError(f"Invalid parameter range: {name}")
            for row in range(count):
                parameters[row][name] = float(limits[0] + unit[row, column] * (limits[1] - limits[0]))
    digest, code_digest = config_hash(cfg), source_hash()
    chosen = np.rint(np.linspace(0, len(grid) - 1, n_steady)).astype(int)
    forcings = []
    for i in range(count):
        if i < n_steady:
            forcing = {"kind": "constant", "amplitude": float(grid[chosen[i]])}
        else:
            pair = ends if (i - n_steady) % 2 == 0 else ends[::-1]
            forcing = {"kind": "ramp", "start_amplitude": float(pair[0]), "end_amplitude": float(pair[1]),
                       "start_time": float(start_fraction * actual_duration), "end_time": float(end_fraction * actual_duration)}
        forcings.append(forcing)
    # One vectorized forcing callable handles both steady and ramp trajectories.
    omega = np.array([p["omega"] for p in parameters])
    start_amp = np.array([f.get("amplitude", f.get("start_amplitude")) for f in forcings])
    end_amp = np.array([f.get("amplitude", f.get("end_amplitude")) for f in forcings])
    ramp_start, ramp_end = start_fraction * actual_duration, end_fraction * actual_duration
    def e_fn(time):
        fraction = np.clip((time - ramp_start) / (ramp_end - ramp_start), 0.0, 1.0)
        amplitude = start_amp + fraction * (end_amp - start_amp)
        return adapter.ext_signals(time, {"omega": omega}, {"forcing": {"kind": "constant", "amplitude": amplitude}})
    metadata = [{"config_hash": digest, "source_hash": code_digest, "operating_conditions": forcing,
                 "requested_duration_s": duration} for forcing in forcings]
    runs = simulate_batch(
        adapter, parameters, x0, e_fn, None, n_steps, spec=spec,
        substeps=cfg["sampling"].get("substeps", 5), seed=seed,
        run_ids=[f"duffing-{digest[:12]}-{i:04d}" for i in range(count)], metadata=metadata,
    )
    label_cfg = cfg.get("labeling", {})
    setting_names = {item.name for item in fields(LyapunovSettings)}
    unknown = set(label_cfg) - setting_names - {"threshold", "stride_periods"}
    if unknown:
        raise ValueError(f"Unknown labeling settings: {sorted(unknown)}")
    settings = LyapunovSettings(**{key: value for key, value in label_cfg.items() if key in setting_names})
    label_runs(runs, adapter, spec=spec, settings=settings, stride_periods=label_cfg.get("stride_periods", 1))
    validate_dataset(runs, spec, expected_config_hash=digest)
    return spec, runs
