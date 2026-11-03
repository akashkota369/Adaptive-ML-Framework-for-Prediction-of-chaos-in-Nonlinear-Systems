"""Contract serialization, dimensionality, persistence and hash regressions."""

from dataclasses import FrozenInstanceError, fields, replace
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from chaosnet.data.schema import RunRecord, SystemSpec


def test_spec_has_exact_architecture_fields(spec):
    assert [item.name for item in fields(spec)] == [
        "name", "state_names", "state_units", "param_names", "ext_names",
        "ctrl_names", "ctrl_bounds", "ctrl_rate_limit", "Ts", "window",
        "horizon_s", "lyap_threshold", "undesirable", "notes",
    ]
    with pytest.raises(FrozenInstanceError):
        spec.name = "changed"


def test_spec_json_round_trip(spec):
    payload = json.loads(json.dumps(spec.to_dict()))
    restored = SystemSpec.from_dict(payload)
    assert restored == spec
    assert isinstance(restored.state_names, tuple)
    assert isinstance(restored.ctrl_bounds[0], tuple)
    assert restored.undesirable == ("chaotic",)


def test_spec_serialization_does_not_share_notes(spec):
    payload = spec.to_dict()
    payload["notes"]["settings"]["first"] = 99
    assert spec.notes["settings"]["first"] == 1
    restored = SystemSpec.from_dict(payload)
    payload["notes"]["settings"]["first"] = 100
    assert restored.notes["settings"]["first"] == 99


def test_spec_deserialization_defaults_and_unknown_fields(spec):
    payload = spec.to_dict()
    del payload["notes"]
    del payload["undesirable"]
    restored = SystemSpec.from_dict(payload)
    assert restored.notes == {}
    assert restored.undesirable == ("chaotic",)
    with pytest.raises(TypeError):
        SystemSpec.from_dict({**payload, "unexpected": 1})
    del payload["Ts"]
    with pytest.raises(TypeError):
        SystemSpec.from_dict(payload)


def test_spec_hash_is_stable_and_sensitive_to_contract_changes(spec):
    reordered = replace(spec, notes={"settings": {"first": 1, "second": 2}, "source": "test"})
    digest = spec.spec_hash()
    assert len(digest) == 64
    assert int(digest, 16) >= 0
    assert digest == spec.spec_hash() == reordered.spec_hash()
    assert digest == SystemSpec.from_dict(spec.to_dict()).spec_hash()
    assert digest != replace(spec, Ts=0.2).spec_hash()
    assert digest != replace(spec, ctrl_bounds=((-1.0, 1.0),)).spec_hash()


def test_hash_rejects_nonfinite_metadata(spec):
    with pytest.raises(ValueError):
        replace(spec, notes={"invalid": float("nan")}).spec_hash()


def test_derived_dimensions_for_two_and_three_state_systems(spec):
    assert (spec.n_state, spec.n_ext, spec.n_ctrl, spec.n_seq) == (2, 3, 1, 6)
    three_state = replace(
        spec, state_names=("x", "y", "z"), state_units=("1", "1", "1"),
        ext_names=("drifting_parameter",),
    )
    assert (three_state.n_state, three_state.n_ext, three_state.n_ctrl, three_state.n_seq) == (3, 1, 1, 5)


def test_runrecord_npz_round_trip(tmp_path):
    original = RunRecord(
        run_id="test-run-01",
        params={"c": 0.3, "k": -1.0},
        t=np.array([0.0, 0.1, 0.2], dtype=np.float64),
        X=np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32),
        ext=np.empty((3, 0), dtype=np.float64),
        u=np.array([[0.1], [-0.2], [0.0]], dtype=np.float64),
        regime=np.array([0, 1, 0], dtype=np.int8),
        lyap=np.array([-0.1, 0.3, -0.2], dtype=np.float64),
        mode="expert_controlled",
        meta={"seed": 17, "integrator": "fake", "result": {"success": True}, "note": "unit \u03bb"},
    )
    path = tmp_path / "run.npz"
    original.save_npz(path)
    restored = RunRecord.load_npz(path)
    assert [item.name for item in fields(restored)] == [
        "run_id", "params", "t", "X", "ext", "u", "regime", "lyap", "mode", "meta",
    ]
    for name in ("run_id", "params", "mode", "meta"):
        assert getattr(restored, name) == getattr(original, name)
    for name in ("t", "X", "ext", "u", "regime", "lyap"):
        before, after = getattr(original, name), getattr(restored, name)
        np.testing.assert_array_equal(after, before)
        assert after.dtype == before.dtype
    with np.load(path, allow_pickle=False) as archive:
        assert all(not archive[name].dtype.hasobject for name in archive.files)
    # Loading releases the archive handle, including on Windows.
    path.unlink()


def test_runrecord_rejects_incomplete_archive(tmp_path):
    path = tmp_path / "incomplete.npz"
    np.savez(path, t=np.array([0.0]))
    with pytest.raises(ValueError, match="missing fields"):
        RunRecord.load_npz(path)


def test_spec_hash_matches_separate_processes(spec):
    code = (
        "import json, sys; "
        "from chaosnet.data.schema import SystemSpec; "
        "print(SystemSpec.from_dict(json.load(sys.stdin)).spec_hash())"
    )
    hashes = []
    for seed in ("1", "98765"):
        result = subprocess.run(
            [sys.executable, "-c", code],
            input=json.dumps(spec.to_dict()), text=True, capture_output=True,
            env={**os.environ, "PYTHONHASHSEED": seed}, check=True,
        )
        hashes.append(result.stdout.strip())
    assert hashes == [spec.spec_hash(), spec.spec_hash()]
