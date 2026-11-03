"""Shared, deterministic JSON encoding for specification and config hashes."""

import json
from typing import Any


def canonical_json(value: Any) -> str:
    """Encode finite JSON data with sorted string keys and no extra whitespace.

    Tuples are encoded as JSON arrays. Non-string mapping keys are rejected
    because converting them to strings could silently change a data contract.
    """
    _check_keys(value)
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )


def _check_keys(value: Any) -> None:
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("JSON object keys must be strings")
        for item in value.values():
            _check_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _check_keys(item)
