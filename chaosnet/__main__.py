"""Command-line entry point: python -m chaosnet smoke."""

import argparse
import json
from importlib.resources import as_file, files
from pathlib import Path

from chaosnet.configs.loader import load_cfg
from chaosnet.data.schema import SystemSpec
from chaosnet.data.validate import validate
from chaosnet.datagen.fake import make_fake_runs


def smoke(config_path: Path | None = None) -> None:
    """Build a sample contract, generate fake runs, and verify their shapes."""
    if config_path is None:
        with as_file(files("chaosnet.configs").joinpath("example.yaml")) as example:
            cfg = load_cfg(example)
    else:
        cfg = load_cfg(config_path)
    spec = SystemSpec.from_dict(cfg["spec"])
    if cfg["n_runs"] < 1:
        raise ValueError("The smoke test requires at least one fake run")
    runs = make_fake_runs(spec, cfg["n_runs"], cfg["seed"])
    validate(runs, spec)
    print("OK")


def main() -> None:
    parser = argparse.ArgumentParser(prog="chaosnet")
    commands = parser.add_subparsers(dest="command", required=True)
    smoke_parser = commands.add_parser("smoke", help="Validate the Stage 0 data interfaces")
    smoke_parser.add_argument("--config", type=Path, help="Override the bundled example YAML")
    generate = commands.add_parser("generate-duffing", help="Generate and validate a labeled Duffing dataset")
    generate.add_argument("--config", type=Path, help="Override the bundled Duffing YAML")
    generate.add_argument("--output", type=Path, help="New dataset directory (default: config data_dir)")
    generate.add_argument("--runs", type=int, help="Number of steady/ramp runs")
    generate.add_argument("--seed", type=int, help="Reproducible experiment seed")
    duration = generate.add_mutually_exclusive_group()
    duration.add_argument("--duration", type=float, help="Simulation duration in time units, rounded up to a sample")
    duration.add_argument("--duration-periods", type=float, help="Duration in nominal forcing periods")
    args = parser.parse_args()
    if args.command == "smoke":
        smoke(args.config)
    elif args.command == "generate-duffing":
        from chaosnet.datagen.dataset import generate_duffing
        if args.config is None:
            with as_file(files("chaosnet.configs").joinpath("duffing_v1.yaml")) as example:
                cfg = load_cfg(example)
        else:
            cfg = load_cfg(args.config)
        if args.runs is not None:
            cfg["experiments"]["n_runs"] = args.runs
        if args.seed is not None:
            cfg["seed"] = args.seed
        if args.duration is not None:
            cfg["experiments"].pop("duration_periods", None)
            cfg["experiments"]["duration_s"] = args.duration
        if args.duration_periods is not None:
            cfg["experiments"].pop("duration_s", None)
            cfg["experiments"]["duration_periods"] = args.duration_periods
        try:
            summary = generate_duffing(cfg, args.output or Path(cfg["data_dir"]))
        except (ValueError, FileExistsError) as exc:
            parser.error(str(exc))
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
