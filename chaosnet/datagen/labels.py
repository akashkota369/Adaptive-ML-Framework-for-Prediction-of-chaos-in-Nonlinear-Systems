"""Twin-trajectory Lyapunov estimates and conservative frozen-forcing labels."""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from chaosnet.datagen.simulate import advance, parameter_batch


@dataclass(frozen=True)
class LyapunovSettings:
    transient_periods: int = 40
    evaluation_periods: int = 100
    min_evaluation_periods: int = 40
    samples_per_period: int = 32
    substeps: int = 5
    epsilon: float = 1e-8
    convergence_tolerance: float = 0.025
    threshold_margin: float = 0.005
    poincare_points: int = 64
    poincare_tolerance: float = 0.01

    def __post_init__(self):
        for name in ("transient_periods", "evaluation_periods", "min_evaluation_periods", "samples_per_period", "substeps", "poincare_points"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == "transient_periods" else 1):
                raise ValueError(f"{name} has an invalid integer value")
        if self.evaluation_periods < 2 or self.samples_per_period < 8:
            raise ValueError("At least two evaluation periods and eight samples per period are required")
        for name in ("epsilon", "convergence_tolerance", "threshold_margin", "poincare_tolerance"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")


@dataclass(frozen=True)
class LyapunovEstimate:
    exponent: float
    first_half: float
    second_half: float
    reliable: bool
    reasons: tuple[str, ...]
    poincare_period: int
    poincare_spread: float
    poincare_status: str


def poincare_check(points: np.ndarray, tolerance: float = 0.01, max_period: int = 8) -> dict[str, Any]:
    """Check recurrence of stroboscopic points; aperiodicity alone is not chaos."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or not np.all(np.isfinite(points)):
        raise ValueError("Poincare points must be a finite (periods, states) array")
    if len(points) < 8:
        return {"period": 0, "spread": 0.0, "status": "insufficient_data"}
    scale = max(1.0, float(np.max(np.linalg.norm(points, axis=1))))
    spread = float(np.sqrt(np.mean(np.sum((points - points.mean(axis=0))**2, axis=1))))
    for period in range(1, min(max_period, len(points) // 3) + 1):
        distances = np.linalg.norm(points[period:] - points[:-period], axis=1) / scale
        if np.quantile(distances, 0.9) <= tolerance:
            return {"period": period, "spread": spread, "status": "periodic"}
    return {"period": 0, "spread": spread, "status": "aperiodic"}


def estimate_lyapunov(adapter, p_batch, x0, amplitudes, *, start_times=0.0,
                      threshold: float = 0.01, settings: LyapunovSettings | None = None) -> list[LyapunovEstimate]:
    """Largest conditional exponent of the periodically forced plant, in 1/time.

    Both twins experience exactly the same forcing phase and zero control.
    Separation is renormalized at every sample, including transient removal.
    Batched trajectories can have different forcing frequencies and phases.
    This estimates a frozen-forcing attractor, not the full ramped system or
    any closed-loop controller's augmented state.
    """
    settings = settings or LyapunovSettings()
    x = np.asarray(x0, dtype=float).copy()
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or x.shape[1] != 2 or len(x) == 0 or not np.all(np.isfinite(x)):
        raise ValueError("Lyapunov x0 must be finite and have shape (batch, 2)")
    batch = len(x)
    p = parameter_batch(p_batch, batch, ("c", "k", "beta", "omega"))
    amplitudes = np.broadcast_to(np.asarray(amplitudes, dtype=float), (batch,))
    start_times = np.broadcast_to(np.asarray(start_times, dtype=float), (batch,))
    if np.any(p["omega"] <= 0) or not np.all(np.isfinite(amplitudes)) or not np.all(np.isfinite(start_times)) or not np.isfinite(threshold):
        raise ValueError("Frequencies must be positive; forcing, times, and threshold must be finite")
    dt = (2 * np.pi / p["omega"]) / settings.samples_per_period
    direction = np.array([1.0, 0.61803398875])
    direction /= np.linalg.norm(direction)
    twin = x + settings.epsilon * direction
    burn = settings.transient_periods * settings.samples_per_period
    count = settings.evaluation_periods * settings.samples_per_period
    split = count // 2
    sums = np.zeros((2, batch))
    failed = np.zeros(batch, dtype=bool)
    sections = []
    # Stacking both twins along a leading axis preserves per-run parameters.
    forcing_cfg = {"forcing": {"kind": "constant", "amplitude": amplitudes}}
    def e_fn(t):
        return adapter.ext_signals(t, p, forcing_cfg)
    zero = np.zeros((batch, 1))
    for step in range(burn + count):
        time = start_times + step * dt
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            try:
                pair = advance(adapter, time, np.stack((x, twin)), p, e_fn, zero, dt, settings.substeps)
            except FloatingPointError as exc:
                raise ValueError("Lyapunov continuation diverged; revise integration settings or parameters") from exc
        x, twin = pair[0], pair[1]
        delta = twin - x
        distance = np.linalg.norm(delta, axis=1)
        bad = ~np.isfinite(distance) | (distance <= 1e-15) | (np.max(np.abs(x), axis=1) > 1e6)
        if not np.all(np.isfinite(x)):
            raise ValueError("Non-finite Lyapunov reference trajectory")
        failed |= bad
        safe_distance = np.where(bad, settings.epsilon, distance)
        delta[bad] = settings.epsilon * direction
        twin = x + delta * (settings.epsilon / safe_distance)[:, None]
        if step >= burn:
            index = step - burn
            sums[int(index >= split)] += np.log(safe_distance / settings.epsilon)
            if (index + 1) % settings.samples_per_period == 0:
                sections.append(x.copy())
    first = sums[0] / (split * dt)
    second = sums[1] / ((count - split) * dt)
    exponent = (sums[0] + sums[1]) / (count * dt)
    section_array = np.asarray(sections[-settings.poincare_points:])
    results = []
    for i in range(batch):
        reasons = []
        if failed[i]:
            reasons.append("separation_failure")
        if settings.evaluation_periods < settings.min_evaluation_periods:
            reasons.append("insufficient_evaluation")
        if settings.transient_periods < 5:
            reasons.append("insufficient_transient")
        if abs(first[i] - second[i]) > settings.convergence_tolerance:
            reasons.append("halves_disagree")
        if abs(exponent[i] - threshold) <= settings.threshold_margin or (first[i] > threshold) != (second[i] > threshold):
            reasons.append("near_threshold")
        cross = poincare_check(section_array[:, i], settings.poincare_tolerance)
        if exponent[i] > threshold and cross["status"] == "periodic":
            reasons.append("poincare_disagreement")
        results.append(LyapunovEstimate(
            exponent=float(exponent[i]), first_half=float(first[i]), second_half=float(second[i]),
            reliable=not reasons, reasons=tuple(reasons), poincare_period=cross["period"],
            poincare_spread=cross["spread"], poincare_status=cross["status"],
        ))
    return results


def label_runs(runs, adapter, *, spec, settings: LyapunovSettings | None = None, stride_periods: int = 1):
    """Populate binary regime/finite Lyapunov arrays and a separate validity mask.

    Ramp labels use frozen-amplitude continuations from the sampled state and
    forcing phase. Between anchors values are held left; intervals joining
    uncertain or conflicting anchors are marked invalid. Fixed forcing uses
    one continuation from the run's initial condition. Controlled runs are
    rejected because their full feedback state is outside Stage 1.
    """
    settings = settings or LyapunovSettings()
    if not isinstance(stride_periods, int) or stride_periods < 1:
        raise ValueError("stride_periods must be a positive integer")
    jobs, anchors_by_run = [], []
    for run in runs:
        if run.mode != "open_loop" or np.any(run.u != 0):
            raise ValueError("Stage 1 frozen-forcing labels support uncontrolled runs only")
        steady = np.allclose(run.ext[:, 0], run.ext[0, 0], rtol=0, atol=1e-14)
        stride = max(1, int(round(stride_periods * 2 * np.pi / run.params["omega"] / spec.Ts)))
        anchors = [0] if steady else sorted(set(range(0, len(run.t), stride)) | {len(run.t) - 1})
        anchors_by_run.append(anchors)
        jobs.extend((run, index) for index in anchors)
    if not jobs:
        return runs
    estimates = estimate_lyapunov(
        adapter, [run.params for run, _ in jobs], np.array([run.X[k] for run, k in jobs]),
        np.array([run.ext[k, 0] for run, k in jobs]),
        start_times=np.array([run.t[k] for run, k in jobs]), threshold=spec.lyap_threshold, settings=settings,
    )
    offset = 0
    for run, anchors in zip(runs, anchors_by_run):
        batch = estimates[offset:offset + len(anchors)]
        offset += len(anchors)
        valid = np.zeros(len(run.t), dtype=bool)
        for j, (index, estimate) in enumerate(zip(anchors, batch)):
            stop = anchors[j + 1] if j + 1 < len(anchors) else len(run.t)
            regime = int(estimate.exponent > spec.lyap_threshold)
            run.lyap[index:stop] = estimate.exponent
            run.regime[index:stop] = regime
            interval_valid = estimate.reliable
            if j + 1 < len(anchors):
                next_estimate = batch[j + 1]
                interval_valid &= next_estimate.reliable and regime == int(next_estimate.exponent > spec.lyap_threshold)
            valid[index:stop] = interval_valid
            valid[index] = estimate.reliable
        run.meta["label_status"] = "labeled"
        run.meta["label_valid"] = valid.tolist()
        run.meta["labeling"] = {
            "method": "benettin_frozen_forcing", "settings": asdict(settings),
            "threshold": spec.lyap_threshold, "stride_periods": stride_periods,
            "interpolation": "left_hold_conflicting_intervals_invalid",
            "anchors": [{"sample_index": k, **asdict(result)} for k, result in zip(anchors, batch)],
        }
    return runs
