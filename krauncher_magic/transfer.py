# Copyright (c) 2026 Ilya Sergeev. Licensed under the MIT License.

"""Namespace transfer — carry notebook values into task kwargs and back.

The generated cell function is exactly what the analyzer classifies and the
worker runs (see codegen.py), so inputs and outputs must be plain JSON-safe
values — the SDK moves task kwargs and results as JSON. Anything that is not
JSON-serializable (a model, a DataFrame, an open handle) is rejected here with
a clear message: move it to a data source / volume. Numeric scalars pass
through so the cas-analyzer's CU estimator still sees ``epochs``/``batch_size``.
"""

from __future__ import annotations

import json
from typing import Any

# Inline payload guard — larger inputs must go through --data / volumes.
MAX_INLINE_MB = 32


class TransferError(ValueError):
    """A value cannot cross the notebook↔task boundary."""


def encode_inputs(names: list[str], user_ns: dict[str, Any]) -> dict[str, Any]:
    """Fetch --in variables from the notebook namespace as task kwargs."""
    kwargs: dict[str, Any] = {}
    for name in names:
        if name not in user_ns:
            raise TransferError(f"--in {name}: not defined in the notebook")
        value = user_ns[name]
        try:
            encoded = json.dumps(value)
        except (TypeError, ValueError) as exc:
            raise TransferError(
                f"--in {name}: {type(value).__name__} is not JSON-serializable "
                f"({exc}). Pass JSON-safe values (numbers, strings, lists, dicts) "
                f"or move large/complex data to a data source / volume."
            ) from exc
        if len(encoded.encode("utf-8")) > MAX_INLINE_MB * 1024 * 1024:
            raise TransferError(
                f"--in {name}: exceeds the {MAX_INLINE_MB} MB inline limit — "
                f"use --data / a volume."
            )
        kwargs[name] = value
    return kwargs


def decode_outputs(output: Any, names: list[str]) -> dict[str, Any]:
    """Pull the task's returned outputs dict back into notebook values."""
    if not isinstance(output, dict):
        raise TransferError(
            f"remote cell returned {type(output).__name__}, expected the "
            f"outputs dict — was the cell body altered?"
        )
    values: dict[str, Any] = {}
    for name in names:
        if name not in output:
            raise TransferError(f"--out {name}: missing from the task result")
        values[name] = output[name]
    return values
