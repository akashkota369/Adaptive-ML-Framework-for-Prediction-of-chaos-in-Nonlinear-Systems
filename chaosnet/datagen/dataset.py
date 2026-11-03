"""Validated NPZ datasets with manifests, configuration, hashes, and summaries."""

from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import yaml

from chaosnet._serialization import canonical_json
from chaosnet.configs.loader import config_hash, load_cfg
from chaosnet.data.schema import RunRecord, SystemSpec
from chaosnet.data.validate import validate_dataset


def summarize_dataset(runs) -> dict:
    """Count reliable labels separately from all uncertain numerical estimates."""
    rows, reasons, poincare = [], Counter(), Counter()
    reliable_anchors = total_anchors = 0
    for run in runs:
        valid = np.asarray(run.meta["label_valid"], dtype=bool)
        anchors = run.meta["labeling"]["anchors"]
        total_anchors += len(anchors)
        reliable_anchors += sum(anchor["reliable"] for anchor in anchors)
        for anchor in anchors:
            reasons.update(anchor["reasons"])
            poincare.update([anchor["poincare_status"]])
        rows.append({
            "run_id": run.run_id, "kind": run.meta["operating_conditions"]["kind"], "samples": len(run.t),
            "reliable_nonchaotic": int(np.count_nonzero(valid & (run.regime == 0))),
            "reliable_chaotic": int(np.count_nonzero(valid & (run.regime == 1))),
            "uncertain": int(np.count_nonzero(~valid)),
            "lyap_min": float(run.lyap.min()), "lyap_max": float(run.lyap.max()),
        })
    totals = {name: sum(row[name] for row in rows) for name in ("samples", "reliable_nonchaotic", "reliable_chaotic", "uncertain")}
    return {
        "n_runs": len(runs), **totals,
        "reliable_fraction": (totals["samples"] - totals["uncertain"]) / totals["samples"] if totals["samples"] else 0.0,
        "label_anchors": total_anchors, "reliable_anchors": reliable_anchors,
        "unreliable_anchor_reasons": dict(sorted(reasons.items())), "poincare_status_counts": dict(sorted(poincare.items())),
        "label_meaning": "asymptotic frozen-forcing regime; use only label_valid samples",
        "runs": rows,
    }


def _file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _manifest_row(run, filename, digest):
    valid = np.asarray(run.meta["label_valid"], dtype=bool)
    return {
        "run_id": run.run_id, "file": filename, "sha256": digest,
        "params": canonical_json(run.params), "operating_conditions": canonical_json(run.meta["operating_conditions"]),
        "n_samples": str(len(run.t)), "duration_s": str(float(run.t[-1] - run.t[0])), "mode": run.mode,
        "seed": str(run.meta["seed"]), "config_hash": run.meta["config_hash"],
        "spec_hash": run.meta["spec_hash"], "source_hash": run.meta["source_hash"],
        "reliable_nonchaotic": str(np.count_nonzero(valid & (run.regime == 0))),
        "reliable_chaotic": str(np.count_nonzero(valid & (run.regime == 1))), "uncertain": str(np.count_nonzero(~valid)),
    }


def save_dataset(path, spec, runs, cfg) -> dict:
    """Validate first, stage complete output, then publish it as a new directory.

    Existing directories are never overwritten. A failed write cannot publish a
    partial dataset. Temporary staging is confined to the destination's parent.
    """
    validate_dataset(runs, spec, expected_config_hash=config_hash(cfg))
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f"Dataset already exists: {path}; choose a new output directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{path.name}-", dir=path.parent))
    try:
        rows = []
        for run in runs:
            filename = f"{run.run_id}.npz"
            run.save_npz(staging / filename)
            rows.append(_manifest_row(run, filename, _file_hash(staging / filename)))
        with (staging / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (staging / "spec.json").write_text(json.dumps(spec.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
        summary = summarize_dataset(runs)
        (staging / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        # Reopen every staged run before making the dataset visible at its final path.
        load_dataset(staging)
        os.rename(staging, path)
        return summary
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def load_dataset(path):
    """Load and revalidate every run, manifest entry, digest, and summary."""
    path = Path(path).resolve()
    spec = SystemSpec.from_dict(json.loads((path / "spec.json").read_text(encoding="utf-8")))
    cfg = load_cfg(path / "config.yaml")
    with (path / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Dataset manifest is empty")
    names = [row.get("file", "") for row in rows]
    if len(set(names)) != len(names) or {file.name for file in path.glob("*.npz")} != set(names):
        raise ValueError("Manifest must list each NPZ run exactly once, with no missing or orphan files")
    runs = []
    for row in rows:
        filename = row["file"]
        file = path / filename
        if Path(filename).name != filename or file.resolve().parent != path or file.suffix != ".npz":
            raise ValueError("Unsafe manifest filename")
        digest = _file_hash(file)
        if digest != row.get("sha256"):
            raise ValueError(f"Run file checksum mismatch: {filename}")
        run = RunRecord.load_npz(file)
        if filename != f"{run.run_id}.npz":
            raise ValueError("Run ID does not match its filename")
        runs.append(run)
    validate_dataset(runs, spec, expected_config_hash=config_hash(cfg))
    for row, run in zip(rows, runs):
        if row != _manifest_row(run, row["file"], row["sha256"]):
            raise ValueError(f"Manifest metadata mismatch: {run.run_id}")
    if json.loads((path / "summary.json").read_text(encoding="utf-8")) != summarize_dataset(runs):
        raise ValueError("Dataset summary disagrees with saved runs")
    return spec, runs


def generate_duffing(cfg, output):
    """Generate and publish a complete Duffing experiment dataset."""
    from chaosnet.datagen.experiments import make_duffing_runs
    if Path(output).exists():
        raise FileExistsError(f"Dataset already exists: {output}; choose a new output directory")
    spec, runs = make_duffing_runs(cfg)
    return save_dataset(output, spec, runs, cfg)
