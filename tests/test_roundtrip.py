# Copyright (c) 2026 Ilya Sergeev. Licensed under the MIT License.

"""Round-trip tests: transfer ↔ codegen ↔ krauncher serializer (no broker).

Run: PYTHONPATH=.:../cas-client pytest tests/
"""

import inspect

import pytest

from krauncher_magic.codegen import build_cell_function
from krauncher_magic.transfer import (
    TransferError,
    decode_outputs,
    encode_inputs,
)

NS = {"epochs": 3, "name": "bert", "data": [1, 2, 3], "ratio": 0.5}
CELL = "total = sum(data) * epochs\nlabel = name.upper()\n"
INS = ["epochs", "name", "data", "ratio"]
OUTS = ["total", "label"]


def test_json_values_pass_raw():
    # JSON-safe values (scalars and lists) pass through unchanged so the
    # analyzer sees them and the clean function runs on them directly.
    assert encode_inputs(INS, NS) == {
        "epochs": 3, "name": "bert", "data": [1, 2, 3], "ratio": 0.5,
    }


def test_roundtrip_through_generated_function():
    kwargs = encode_inputs(INS, NS)
    fn = build_cell_function(CELL, INS, OUTS)
    assert decode_outputs(fn(**kwargs), OUTS) == {"total": 18, "label": "BERT"}


def test_serializer_worker_simulation():
    """krauncher's serialize_function must see the generated source, and the
    serialized string must execute standalone (as the worker does)."""
    krauncher = pytest.importorskip("krauncher.serializer")
    fn = build_cell_function(CELL, INS, OUTS)
    code_string, entry = krauncher.serialize_function(fn)
    assert entry == "_kr_cell"
    ns: dict = {}
    exec(compile(code_string, "<worker>", "exec"), ns)  # noqa: S102
    kwargs = encode_inputs(INS, NS)
    assert decode_outputs(ns[entry](**kwargs), OUTS) == {"total": 18, "label": "BERT"}


def test_generated_source_is_plain_user_code():
    """What the analyzer classifies and the worker runs must be the cell body,
    with no transport scaffolding leaking in."""
    src = inspect.getsource(build_cell_function(CELL, INS, OUTS))
    for token in ("pickle", "base64", "_kr_dec", "_kr_enc", "b64"):
        assert token not in src


def test_toplevel_global_rejected():
    with pytest.raises(TransferError):
        build_cell_function("global x\nx = 1", [], ["x"])


def test_nested_global_allowed():
    build_cell_function("def f():\n    global q\n    q = 1\nf()", [], [])


def test_non_json_input_rejected():
    with pytest.raises(TransferError):
        encode_inputs(["f"], {"f": lambda: 1})


def test_missing_input_rejected():
    with pytest.raises(TransferError):
        encode_inputs(["nope"], {})
