"""Load safe YAML mappings and hash their content independently of formatting."""

import hashlib
from pathlib import Path
from typing import Any, Mapping

import yaml

from chaosnet._serialization import canonical_json


def load_cfg(path: str | Path) -> dict[str, Any]:
    """Read a YAML mapping containing finite JSON-compatible configuration data."""
    with Path(path).open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a YAML mapping")
    canonical_json(cfg)
    return cfg


def config_hash(cfg: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 digest of the parsed configuration content."""
    return hashlib.sha256(canonical_json(dict(cfg)).encode("utf-8")).hexdigest()
