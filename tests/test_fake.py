"""Fake runs must be reproducible, independent, and conform to both widths."""

from dataclasses import replace

import numpy as np
import pytest

from chaosnet.data.validate import validate
from chaosnet.datagen.fake import make_fake_runs


def test_fake_runs_are_reproducible(spec):
    first = make_fake_runs(spec, n_runs=3, seed=13)
    second = make_fake_runs(spec, n_runs=3, seed=13)
    for left, right in zip(first, second):
        assert left.run_id == right.run_id
        assert left.params == right.params
        assert left.meta == right.meta
        assert left.mode == right.mode == "open_loop"
        for field in ("t", "X", "ext", "u", "regime", "lyap"):
            np.testing.assert_array_equal(getattr(left, field), getattr(right, field))
    assert len({run.run_id for run in first}) == 3
    assert not np.shares_memory(first[0].X, first[1].X)
    assert not np.array_equal(first[0].X, make_fake_runs(spec, 1, seed=14)[0].X)


def test_fake_runs_do_not_change_global_rng(spec):
    before = np.random.get_state()
    make_fake_runs(spec, n_runs=2, seed=27)
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


@pytest.mark.parametrize("n_state,n_ext", [(2, 3), (3, 1), (2, 0)])
def test_fake_run_dimensions(spec, n_state, n_ext):
    varied = replace(
        spec,
        state_names=tuple(f"x{i}" for i in range(n_state)),
        state_units=("1",) * n_state,
        ext_names=tuple(f"e{i}" for i in range(n_ext)),
    )
    runs = make_fake_runs(varied, n_runs=2, seed=4)
    report = validate(runs, varied)
    assert report["n_runs"] == 2
    assert report["n_samples"] == sum(len(run.t) for run in runs)
    for run in runs:
        assert run.X.shape == (len(run.t), n_state)
        assert run.ext.shape == (len(run.t), n_ext)
        assert run.u.shape == (len(run.t), varied.n_ctrl)
        assert run.regime.dtype == np.int8
        assert run.meta["synthetic"] is True


def test_empty_fake_dataset_and_invalid_count(spec):
    assert make_fake_runs(spec, 0, seed=3) == []
    with pytest.raises(ValueError, match="n_runs"):
        make_fake_runs(spec, -1, seed=3)


def test_validation_rejects_wrong_shape_and_duplicate_ids(spec):
    run = make_fake_runs(spec, 1, seed=3)[0]
    with pytest.raises(ValueError, match="Duplicate run_id"):
        validate([run, run], spec)
    run.u = np.zeros((len(run.t), 2))
    with pytest.raises(ValueError, match="u must have shape"):
        validate([run], spec)


def test_validation_rejects_nonfinite_data_and_wrong_sampling(spec):
    run = make_fake_runs(spec, 1, seed=3)[0]
    run.X[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate([run], spec)
    run.X[0, 0] = 0.0
    run.t[2] += spec.Ts / 2
    with pytest.raises(ValueError, match="timestamps"):
        validate([run], spec)
