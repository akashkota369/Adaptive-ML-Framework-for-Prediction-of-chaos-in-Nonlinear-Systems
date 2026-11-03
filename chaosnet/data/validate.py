"""Stage 0 checks for run shapes, sampling, names, and finite values."""

from collections.abc import Sequence
import math

import numpy as np

from chaosnet.data.schema import RunRecord, SystemSpec


def validate(runs: Sequence[RunRecord], spec: SystemSpec) -> dict[str, int]:
    """Raise ValueError on a contract violation; otherwise return run counts.

    These checks establish the shared data interface. They do not validate
    physical equations, chaos labels, or controller effectiveness.
    """
    if not np.isfinite(spec.Ts) or spec.Ts <= 0:
        raise ValueError("SystemSpec.Ts must be positive and finite")
    if not isinstance(spec.window, int) or spec.window < 1:
        raise ValueError("SystemSpec.window must be a positive integer")
    if not np.isfinite(spec.horizon_s) or spec.horizon_s < 0:
        raise ValueError("SystemSpec.horizon_s must be non-negative and finite")
    if len(spec.state_units) != spec.n_state:
        raise ValueError("state_units must match state_names")
    if len(spec.ctrl_bounds) != spec.n_ctrl or len(spec.ctrl_rate_limit) != spec.n_ctrl:
        raise ValueError("Control constraints must match ctrl_names")

    seen: set[str] = set()
    total_samples = 0
    minimum_samples = spec.window + math.ceil(spec.horizon_s / spec.Ts)
    for run in runs:
        if run.run_id in seen:
            raise ValueError(f"Duplicate run_id: {run.run_id}")
        seen.add(run.run_id)
        if not isinstance(run.t, np.ndarray) or run.t.ndim != 1:
            raise ValueError(f"{run.run_id}: t must have shape (T,)")
        length = len(run.t)
        if length < minimum_samples:
            raise ValueError(f"{run.run_id}: run is too short for window and horizon")
        expected = {
            "t": (length,), "X": (length, spec.n_state),
            "ext": (length, spec.n_ext), "u": (length, spec.n_ctrl),
            "regime": (length,), "lyap": (length,),
        }
        for name, shape in expected.items():
            values = getattr(run, name)
            if not isinstance(values, np.ndarray) or values.shape != shape:
                raise ValueError(f"{run.run_id}: {name} must have shape {shape}")
            if values.dtype.kind not in "biuf" or not np.all(np.isfinite(values)):
                raise ValueError(f"{run.run_id}: {name} must contain finite real values")
        if not np.allclose(np.diff(run.t), spec.Ts, rtol=1e-6, atol=spec.Ts * 1e-8):
            raise ValueError(f"{run.run_id}: timestamps must be spaced by Ts")
        if set(run.params) != set(spec.param_names):
            raise ValueError(f"{run.run_id}: parameter names do not match SystemSpec")
        if any(not np.isscalar(value) or not np.isfinite(value) for value in run.params.values()):
            raise ValueError(f"{run.run_id}: parameters must be finite scalars")
        if run.mode not in {"open_loop", "expert_controlled", "candidate_trial"}:
            raise ValueError(f"{run.run_id}: unknown run mode {run.mode!r}")
        total_samples += length
    return {"n_runs": len(runs), "n_samples": total_samples}


def validate_dataset(runs: Sequence[RunRecord], spec: SystemSpec, *, expected_config_hash: str | None = None) -> dict[str, int]:
    """Strict persistence gate for Stage 1 scientific datasets.

    The Stage 0 validator and contracts remain unchanged. Unlabeled simulation
    output is intentionally rejected here, as are inconsistent reliability
    masks, actuator violations, and incomplete provenance.
    """
    import re
    from chaosnet._serialization import canonical_json
    from chaosnet.systems.duffing import DuffingAdapter

    report = validate(runs, spec)
    if not runs:
        raise ValueError("A dataset must contain at least one run")
    bounds = np.asarray(spec.ctrl_bounds, dtype=float)
    if bounds.shape != (spec.n_ctrl, 2) or not np.all(np.isfinite(bounds)) or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("Invalid actuator bounds in specification")
    for run in runs:
        prefix = f"{run.run_id}: "
        if not isinstance(run.run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run.run_id):
            raise ValueError("run_id must be a safe, nonempty filename component")
        if np.any(run.u < bounds[:, 0] - 1e-12) or np.any(run.u > bounds[:, 1] + 1e-12):
            raise ValueError(prefix + "actuator bounds violated")
        if np.any(np.abs(np.diff(run.u, axis=0)) > np.asarray(spec.ctrl_rate_limit) * spec.Ts + 1e-12):
            raise ValueError(prefix + "actuator rate limit violated")
        if not isinstance(run.meta, dict):
            raise ValueError(prefix + "metadata must be a dictionary")
        try:
            canonical_json(run.meta)
        except (ValueError, TypeError) as exc:
            raise ValueError(prefix + "metadata must be finite JSON data") from exc
        meta = run.meta
        required = {"synthetic", "seed", "spec_hash", "config_hash", "source_hash", "integrator",
                    "simulation", "initial_state", "operating_conditions", "label_status", "label_valid", "labeling"}
        if not required <= meta.keys():
            raise ValueError(prefix + f"incomplete metadata: {sorted(required - meta.keys())}")
        if meta["synthetic"] is not False or meta["label_status"] != "labeled":
            raise ValueError(prefix + "only real, labeled simulations can enter this dataset")
        if run.mode != "open_loop" or np.any(run.u != 0):
            raise ValueError(prefix + "frozen-forcing labels do not describe controlled trajectories")
        if isinstance(meta["seed"], bool) or not isinstance(meta["seed"], int) or meta["seed"] < 0:
            raise ValueError(prefix + "invalid seed")
        for name in ("spec_hash", "config_hash", "source_hash"):
            if not isinstance(meta[name], str) or not re.fullmatch(r"[0-9a-f]{64}", meta[name]):
                raise ValueError(prefix + f"invalid {name}")
        if meta["spec_hash"] != spec.spec_hash() or (expected_config_hash is not None and meta["config_hash"] != expected_config_hash):
            raise ValueError(prefix + "specification or configuration hash mismatch")
        integrator = meta["integrator"]
        if not isinstance(integrator, dict) or integrator.get("name") != "rk4" or integrator.get("Ts") != spec.Ts or not isinstance(integrator.get("substeps"), int) or integrator["substeps"] < 1:
            raise ValueError(prefix + "invalid integrator metadata")
        simulation = meta["simulation"]
        if not isinstance(simulation, dict) or simulation.get("n_steps") != len(run.t) - 1 or simulation.get("t0") != run.t[0] or not np.isclose(simulation.get("duration_s", -1), run.t[-1] - run.t[0]):
            raise ValueError(prefix + "incomplete simulation or mismatched duration")
        if np.asarray(meta["initial_state"]).shape != (spec.n_state,) or not np.array_equal(meta["initial_state"], run.X[0]):
            raise ValueError(prefix + "initial state does not match trajectory")
        operating = meta["operating_conditions"]
        if not isinstance(operating, dict) or operating.get("kind") not in {"constant", "ramp"}:
            raise ValueError(prefix + "invalid operating conditions")
        if spec.name == "duffing_v1":
            try:
                expected_ext = DuffingAdapter().ext_signals(run.t, run.params, {"forcing": operating})
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(prefix + "incomplete forcing metadata") from exc
            if not np.allclose(run.ext, expected_ext, rtol=1e-12, atol=1e-12):
                raise ValueError(prefix + "external signals disagree with forcing metadata")
        mask = np.asarray(meta["label_valid"])
        if mask.shape != run.t.shape or mask.dtype != np.bool_:
            raise ValueError(prefix + "label_valid must be a boolean per sample")
        if run.regime.dtype != np.int8 or not np.all(np.isin(run.regime, [0, 1])):
            raise ValueError(prefix + "regime must be a binary int8 array")
        if not np.array_equal(run.regime, (run.lyap > spec.lyap_threshold).astype(np.int8)):
            raise ValueError(prefix + "regime does not match the Lyapunov threshold")
        labeling = meta["labeling"]
        if not isinstance(labeling, dict) or labeling.get("method") != "benettin_frozen_forcing" or labeling.get("threshold") != spec.lyap_threshold or not isinstance(labeling.get("settings"), dict):
            raise ValueError(prefix + "invalid labeling provenance")
        anchors = labeling.get("anchors")
        if not isinstance(anchors, list) or not anchors:
            raise ValueError(prefix + "missing label anchors")
        indices = [anchor.get("sample_index") for anchor in anchors]
        if any(not isinstance(index, int) or not 0 <= index < len(run.t) for index in indices) or indices[0] != 0 or indices != sorted(set(indices)):
            raise ValueError(prefix + "invalid label anchor indices")
        if operating["kind"] == "ramp" and indices[-1] != len(run.t) - 1:
            raise ValueError(prefix + "ramp labeling does not cover the final sample")
        expected_mask = np.zeros(len(run.t), dtype=bool)
        for j, anchor in enumerate(anchors):
            index = indices[j]
            stop = indices[j + 1] if j + 1 < len(indices) else len(run.t)
            if not isinstance(anchor.get("reliable"), bool) or not isinstance(anchor.get("reasons"), (list, tuple)) or anchor["reliable"] != (not anchor["reasons"]):
                raise ValueError(prefix + "inconsistent anchor reliability")
            if any(not isinstance(anchor.get(name), (int, float)) or not np.isfinite(anchor[name]) for name in ("exponent", "first_half", "second_half", "poincare_spread")):
                raise ValueError(prefix + "non-finite or incomplete Lyapunov anchor")
            if not np.all(run.lyap[index:stop] == anchor["exponent"]):
                raise ValueError(prefix + "stored Lyapunov array disagrees with anchors")
            interval_valid = anchor["reliable"]
            if j + 1 < len(anchors):
                following = anchors[j + 1]
                interval_valid &= following.get("reliable") is True and (anchor["exponent"] > spec.lyap_threshold) == (following["exponent"] > spec.lyap_threshold)
            expected_mask[index:stop] = interval_valid
            expected_mask[index] = anchor["reliable"]
        if not np.array_equal(mask, expected_mask):
            raise ValueError(prefix + "label validity mask disagrees with anchor reliability")
    return report
