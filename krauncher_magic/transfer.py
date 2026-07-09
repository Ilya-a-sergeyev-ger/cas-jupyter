# Copyright (c) 2026 Ilya Sergeev. Licensed under the MIT License.

"""Namespace transfer — encode notebook values into task kwargs and back.

Task args travel as JSON, so every value must be JSON-representable:

- JSON-safe scalars (int/float/str/bool/None) pass through raw — this keeps
  numeric args (epochs, batch_size, ...) visible to the cas-analyzer CU
  estimator, exactly like hand-written ``@client.task`` kwargs.
- Everything else is pickled + base64-encoded into a tagged string that the
  generated wrapper decodes on the worker (see codegen.py).

Outputs come back as a ``{name: tagged-value}`` dict built by the wrapper and
are decoded here before injection into the notebook namespace.
"""

from __future__ import annotations

import base64
import pickle
from typing import Any

# Tag prefixing pickled values so raw strings can never be mistaken for them.
PICKLE_TAG = "__krpkl__:"

# Inline payload guard — larger inputs must go through --data / volumes.
MAX_INLINE_MB = 32

_JSON_SCALARS = (int, float, str, bool, type(None))


class TransferError(ValueError):
    """A value cannot cross the notebook↔task boundary."""


def encode_inputs(names: list[str], user_ns: dict[str, Any]) -> dict[str, Any]:
    """Fetch --in variables from the notebook namespace as task kwargs."""
    kwargs: dict[str, Any] = {}
    for name in names:
        if name not in user_ns:
            raise TransferError(f"--in {name}: not defined in the notebook")
        value = user_ns[name]
        if isinstance(value, _JSON_SCALARS) and not (
            isinstance(value, str) and value.startswith(PICKLE_TAG)
        ):
            kwargs[name] = value
            continue
        try:
            blob = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            raise TransferError(
                f"--in {name}: not picklable ({exc}). Functions/classes defined "
                f"in the notebook cannot transfer in Phase 0 — move the "
                f"definition into the cell."
            ) from exc
        if len(blob) > MAX_INLINE_MB * 1024 * 1024:
            raise TransferError(
                f"--in {name}: {len(blob) / 1e6:.0f} MB exceeds the "
                f"{MAX_INLINE_MB} MB inline limit — use --data / a volume."
            )
        kwargs[name] = PICKLE_TAG + base64.b64encode(blob).decode("ascii")
    return kwargs


def decode_outputs(output: Any, names: list[str]) -> dict[str, Any]:
    """Decode the wrapper's return dict back into notebook values."""
    if not isinstance(output, dict):
        raise TransferError(
            f"remote cell returned {type(output).__name__}, expected the "
            f"outputs dict — was the cell body altered?"
        )
    values: dict[str, Any] = {}
    for name in names:
        if name not in output:
            raise TransferError(f"--out {name}: missing from the task result")
        raw = output[name]
        if isinstance(raw, str) and raw.startswith(PICKLE_TAG):
            blob = base64.b64decode(raw[len(PICKLE_TAG):])
            values[name] = pickle.loads(blob)
        else:
            values[name] = raw
    return values
