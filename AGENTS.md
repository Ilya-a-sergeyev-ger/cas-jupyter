# AGENTS.md — instructions for coding agents

## What this repo is

`krauncher_magic` — the `%%krauncher` IPython cell magic. It wraps a notebook
cell into a function and submits it through the Krauncher SDK
(`@client.task`) to the CaS broker, which picks the cheapest suitable GPU.
This repo is the **notebook front-end only**: no broker, scheduling, billing
or analyzer logic belongs here — that lives in the CaS monorepo, of which
this repo is a submodule (sibling `cas-client` = the `krauncher` SDK).

Design docs are internal and live in the monorepo: `doc/jupyter_architecture.md`
(read before touching transfer/codegen) and `doc/jupyter_roadmap.md` (defines
the phase boundaries). Do not add strategy, positioning or marketing material
to this repo — it is public and user-facing; internal docs belong in the
monorepo `doc/`.

## Hard invariants — do not break

1. **Mirror contract.** `codegen.py` inlines the decode/encode logic into the
   generated wrapper because `krauncher_magic` is NOT installed on the worker.
   The inlined prologue/epilogue must stay byte-compatible with
   `transfer.PICKLE_TAG` and the tagged-string format. Changing one side
   without the other silently corrupts values crossing the boundary — the
   round-trip tests exist to catch this; keep them passing.
2. **`inspect.getsource` must see generated functions.** The SDK serializer
   works via `inspect.getsource`, so every generated wrapper is registered in
   `linecache` under a unique pseudo-filename before `exec`. Any new codegen
   path must do the same.
3. **JSON scalars pass raw.** `int/float/str/bool/None` inputs are NOT
   pickled — the cas-analyzer extracts numeric kwargs (epochs, batch_size…)
   for its CU estimate. Pickling them would blind the estimator.
4. **No IPython import at package top level.** `transfer.py` / `codegen.py`
   run in test and worker-simulation contexts without IPython; the magic is
   imported lazily inside `load_ipython_extension`.
5. **Per-cell = ephemeral.** Do not add session state, persistent kernels, or
   anything that holds a GPU between cells — the per-task model is the
   product thesis, not an implementation accident.

## Phase discipline

Current state: implemented and documented in the README — auto namespace
detection (free vars → inputs, assigned → outputs, with `--in`/`--out`
overrides), quote-before-run, live streaming, HuggingFace/S3 pre-fetch, and
`--async` non-blocking cells. Do not build ahead of the roadmap (monorepo
`doc/jupyter_roadmap.md`) without being asked — in particular no cloudpickle
(transferred values stay JSON-safe only, hard invariant 3) and nothing beyond
the feature set above unless the task explicitly targets the next phase.

## Engineering conventions

- Simplicity first: if 200 lines could be 50, rewrite. No abstractions for
  single-use code. No features beyond what was asked.
- Surgical changes: don't refactor or "improve" adjacent code, comments, or
  formatting that the task doesn't touch.
- Code comments and documentation in English only.
- Don't rename variables or change interfaces without necessity.
- Don't commit or push unless explicitly asked.

## Testing

```bash
PYTHONPATH=.:../cas-client python -m pytest tests/
```

Tests must not require IPython, a broker, or network. The
worker-simulation test (`exec` of the serialized code string) is the
canary for both hard invariants 1 and 2 — never delete or skip it. If the
`krauncher` SDK is unavailable, tests that need it `importorskip`.

What cannot be tested here: the magic class in a live kernel and the e2e
broker flow — call that out in your report instead of claiming full coverage.
