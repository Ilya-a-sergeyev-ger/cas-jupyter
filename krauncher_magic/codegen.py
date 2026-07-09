# Copyright (c) 2026 Ilya Sergeev. Licensed under the MIT License.

"""Synthesize the remote wrapper function from a notebook cell.

The cell body becomes the body of a generated function whose parameters are
the --in names and whose return value is the ``{name: tagged-value}`` outputs
dict (see transfer.py). The generated source is registered in ``linecache`` so
``inspect.getsource`` — which krauncher's serializer relies on — works on the
exec'd function object.

Mirror contract: the decode/encode prologue+epilogue emitted here must match
transfer.PICKLE_TAG and the tagged-string format exactly — the wrapper runs on
the worker where krauncher_magic is not installed, so the logic is inlined.
"""

from __future__ import annotations

import ast
import linecache
import textwrap
from itertools import count
from typing import Callable

from .transfer import PICKLE_TAG, TransferError

_ENTRY = "_kr_cell"
_seq = count(1)

# JSON-safe scalars mirror (worker side has no krauncher_magic import).
_PROLOGUE = f"""\
def {_ENTRY}({{params}}):
    import base64 as _kr_b64, pickle as _kr_pkl
    _kr_TAG = {PICKLE_TAG!r}
    def _kr_dec(v):
        if isinstance(v, str) and v.startswith(_kr_TAG):
            return _kr_pkl.loads(_kr_b64.b64decode(v[len(_kr_TAG):]))
        return v
    def _kr_enc(v):
        if isinstance(v, (int, float, bool, type(None))):
            return v
        if isinstance(v, str) and not v.startswith(_kr_TAG):
            return v
        return _kr_TAG + _kr_b64.b64encode(
            _kr_pkl.dumps(v, protocol=_kr_pkl.HIGHEST_PROTOCOL)).decode("ascii")
{{decodes}}
"""

_EPILOGUE = """\
    return {{{returns}}}
"""


def _reject_unsupported(cell: str) -> None:
    """Cell bodies run inside a function — reject constructs that break there."""
    try:
        tree = ast.parse(cell)
    except SyntaxError as exc:
        raise TransferError(f"cell does not parse: {exc}") from exc
    # Only top-level statements matter: `return`/`nonlocal` at module level are
    # already SyntaxErrors, and global/nonlocal inside functions defined by the
    # cell are legitimate. A top-level `global` would silently detach the name
    # from the wrapper's locals and break output capture — reject it.
    for node in tree.body:
        if isinstance(node, ast.Global):
            raise TransferError(
                "top-level `global` is not supported inside a %%krauncher cell "
                "(the cell body runs as a function)."
            )


def build_cell_function(cell: str, inputs: list[str], outputs: list[str]) -> Callable:
    """Generate, exec and return the wrapper function for a cell."""
    _reject_unsupported(cell)

    params = ", ".join(f"{n}=None" for n in inputs)
    decodes = "\n".join(f"    {n} = _kr_dec({n})" for n in inputs) or "    pass"
    body = textwrap.indent(cell.rstrip() + "\n", "    ")
    returns = ", ".join(f"{n!r}: _kr_enc({n})" for n in outputs)

    source = (
        _PROLOGUE.format(params=params, decodes=decodes)
        + body
        + _EPILOGUE.format(returns=returns)
    )

    # Register in linecache under a unique pseudo-filename so
    # inspect.getsource (used by krauncher's serializer) sees the code.
    filename = f"<krauncher-cell-{next(_seq)}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)

    namespace: dict = {}
    exec(compile(source, filename, "exec"), namespace)  # noqa: S102
    return namespace[_ENTRY]
