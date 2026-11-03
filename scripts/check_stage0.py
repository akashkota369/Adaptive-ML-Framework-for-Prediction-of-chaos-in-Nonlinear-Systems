"""Run the Stage 0 checks using the active, installed Python 3.11 environment."""

import importlib
import importlib.metadata
from pathlib import Path
import subprocess
import sys


def main() -> None:
    if sys.version_info[:2] != (3, 11):
        raise SystemExit("Stage 0 verification requires Python 3.11")
    if sys.prefix == sys.base_prefix:
        raise SystemExit("Run this script inside the clean Stage 0 virtual environment")

    import torch

    if torch.version.cuda is not None:
        raise SystemExit("A CPU PyTorch build is required")
    print(f"Python {sys.version.split()[0]}; virtual environment: {sys.prefix}", flush=True)
    dependencies = {
        "numpy": "numpy", "scipy": "scipy", "torch": "torch", "pandas": "pandas",
        "scikit-learn": "sklearn", "PyYAML": "yaml", "matplotlib": "matplotlib",
        "pytest": "pytest",
    }
    for name, module in dependencies.items():
        importlib.import_module(module)
        print(f"{name}=={importlib.metadata.version(name)}", flush=True)
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, "-m", "pip", "check"], cwd=root, check=True)
    subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=root, check=True)
    subprocess.run([sys.executable, "-m", "chaosnet", "smoke"], cwd=root, check=True)


if __name__ == "__main__":
    main()
