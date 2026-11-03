"""Exercise the actual installed command from outside the repository."""

import os
from pathlib import Path
import subprocess
import sys


def test_smoke_uses_packaged_configuration(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "chaosnet", "smoke"],
        cwd=tmp_path, capture_output=True, text=True, check=True,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
    )
    assert result.stdout.strip() == "OK"
    assert result.stderr == ""


def test_smoke_accepts_explicit_configuration():
    config = Path(__file__).resolve().parents[1] / "chaosnet" / "configs" / "example.yaml"
    result = subprocess.run(
        [sys.executable, "-m", "chaosnet", "smoke", "--config", str(config)],
        capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == "OK"
