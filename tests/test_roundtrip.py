# Copyright (c) 2026 Ilya Sergeev. Licensed under the MIT License.

"""Round-trip tests: transfer ↔ codegen ↔ krauncher serializer (no broker).

Run: PYTHONPATH=.:../cas-client pytest tests/
"""

import pytest

from krauncher_magic.codegen import build_cell_function
from krauncher_magic.transfer import (
    PICKLE_TAG,
    TransferError,
    decode_outputs,
    encode_inputs,
)

NS = {"epochs": 3, "name": "bert", "data": [1, 2, 3], "ratio": 0.5}
CELL = "total = sum(data) * epochs\nlabel = name.upper()\n"
INS = ["epochs", "name", "data", "ratio"]
OUTS = ["total", "label"]


def test_scalars_pass_raw_objects_pickle():
    kwargs = encode_inputs(INS, NS)
    # JSON scalars stay raw — keeps numeric args visible to the analyzer.
    assert kwargs["epochs"] == 3
    assert kwargs["name"] == "bert"
    assert isinstance(kwargs["data"], str) and kwargs["data"].startswith(PICKLE_TAG)


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


def test_toplevel_global_rejected():
    with pytest.raises(TransferError):
        build_cell_function("global x\nx = 1", [], ["x"])


def test_nested_global_allowed():
    build_cell_function("def f():\n    global q\n    q = 1\nf()", [], [])


def test_unpicklable_input_rejected():
    with pytest.raises(TransferError):
        encode_inputs(["f"], {"f": lambda: 1})


def test_missing_input_rejected():
    with pytest.raises(TransferError):
        encode_inputs(["nope"], {})
