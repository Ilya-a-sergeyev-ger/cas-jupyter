# krauncher-jupyter

`%%krauncher` — a Jupyter cell magic that runs a marked notebook cell as an
ephemeral task on the cheapest suitable remote GPU, via the
[Krauncher](https://krauncher.com) broker. The notebook kernel stays local;
only cells you mark go remote. Price is estimated **before** the cell runs.

```python
%load_ext krauncher_magic
```

```python
%%krauncher --in df,epochs --out model,accuracy --pip torch,transformers
model = train(df, epochs)
accuracy = evaluate(model, df)
```

```
krauncher: estimate — ai_training, ≥16 GB VRAM, 214.0 CU, ~118s (ref)
krauncher: done on RTX A4000 in 131.2s — 3.42 KU → model, accuracy
```

`model` and `accuracy` are now regular variables in the notebook, produced on
a GPU picked for the lowest price for this specific workload.

## Why per-cell

Each marked cell is one ephemeral task: no GPU session, no idle time billed,
no instance to choose. The broker classifies the cell's code (VRAM, workload
class, compute units), quotes it, and dispatches it to the cheapest GPU that
fits — across providers. Unmarked cells cost nothing.

The deliberate trade-off: cells are **cold** — no GPU state persists between
them. Load-model-in-one-cell, use-in-the-next does not work; group related
work into a single cell instead.

## Install

```bash
pip install ipython  # or any Jupyter environment
pip install -e .                 # this package
pip install -e path/to/krauncher # the Krauncher SDK (not on PyPI yet)
```

Credentials — same as the SDK: `CAS_API_KEY` (and optionally
`CAS_BROKER_URL`) as environment variables or in a `.env` next to the
notebook.

## Options

| Flag | Meaning |
|---|---|
| `--in NAMES` | comma-separated notebook variables sent to the task (repeatable) |
| `--out NAMES` | comma-separated variables returned into the notebook (repeatable) |
| `--pip PKGS` | pip packages installed in the sandbox before execution |
| `--vram N` | minimum GPU VRAM in GB (default: auto-classified from the code) |
| `--timeout N` | execution timeout in seconds (default 600) |
| `--gpu-name S` | pin a GPU model (case-insensitive substring, e.g. `A4000`) |
| `--estimate` | classify and print the quote only — do not run |

## How it works

The cell body is wrapped into a generated function: `--in` names become
parameters, `--out` names become the return value. The wrapper is submitted
through the standard Krauncher `@client.task` path — analyzer classification,
cheapest-GPU dispatch, per-task billing — and the outputs are injected back
into the notebook namespace when the task completes.

Transfer rules:

- JSON-safe scalars (`int`/`float`/`str`/`bool`/`None`) pass through raw, so
  numeric arguments like `epochs` stay visible to the analyzer's estimator.
- Everything else is pickled (stdlib) + base64. Values that cannot pickle —
  functions or classes defined in the notebook, open handles — are rejected
  with a clear error: move the definition into the cell.
- Inline transfer is capped at 32 MB per input; larger data belongs in a
  data source / volume (see the Krauncher SDK).

## Current limitations (Phase 0)

- `--in` / `--out` are explicit — no automatic detection of the transfer set.
- stdlib pickle only (no cloudpickle): plain data transfers, notebook-defined
  functions/classes do not.
- No live output streaming during execution; stdout is printed on completion.
- Top-level `global` in a cell is not supported (the body runs as a function).

Planned next: auto namespace detection, live stdout/metrics streaming,
large-data routing through volumes, warm-host affinity.

## Development

```bash
PYTHONPATH=.:../cas-client python -m pytest tests/
```

This repo is a submodule of the CaS monorepo; design docs and the broker-side
architecture live there.

## License

MIT © 2026 Ilya Sergeev
