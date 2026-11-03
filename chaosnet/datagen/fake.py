"""Shape-correct fake data, with no physical simulation or scientific labels."""

import math

import numpy as np

from chaosnet.data.schema import RunRecord, SystemSpec
from chaosnet.data.validate import validate


def make_fake_runs(spec: SystemSpec, n_runs: int, seed: int) -> list[RunRecord]:
    """Generate deterministic runs without touching NumPy's global RNG state.

    Each run is long enough for a full input window and prediction horizon.
    States, external signals and regimes are synthetic; controls are zero.
    The same specification, count, seed and pinned NumPy version reproduce
    every field. These runs are for integration development only.
    """
    if isinstance(n_runs, bool) or not isinstance(n_runs, int) or n_runs < 0:
        raise ValueError("n_runs must be a non-negative integer")
    validate([], spec)
    rng = np.random.default_rng(seed)
    n_steps = spec.window + math.ceil(spec.horizon_s / spec.Ts) + 1
    spec_id = spec.spec_hash()
    runs = []
    for index in range(n_runs):
        params = {name: float(rng.uniform(0.5, 1.5)) for name in spec.param_names}
        states = rng.standard_normal((n_steps, spec.n_state))
        external = rng.standard_normal((n_steps, spec.n_ext))
        regime = rng.integers(0, 2, size=n_steps, dtype=np.int8)
        margin = rng.uniform(0.05, 0.15, size=n_steps)
        lyap = spec.lyap_threshold + np.where(regime == 1, margin, -margin)
        runs.append(RunRecord(
            run_id=f"fake-{spec_id[:12]}-{seed}-{index:04d}",
            params=params,
            t=np.arange(n_steps, dtype=np.float64) * spec.Ts,
            X=states,
            ext=external,
            u=np.zeros((n_steps, spec.n_ctrl), dtype=np.float64),
            regime=regime,
            lyap=lyap,
            mode="open_loop",
            meta={"synthetic": True, "seed": int(seed), "run_index": index, "spec_hash": spec_id},
        ))
    return runs
