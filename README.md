# krauncher-jupyter

`%%krauncher` — a Jupyter cell magic that runs a marked notebook cell as an
ephemeral task on the cheapest suitable remote GPU, via the
[Krauncher](https://krauncher.com) broker. The notebook kernel stays local;
only cells you mark go remote. Price is quoted **before** the cell runs.

```python
%pip install krauncher-jupyter
%load_ext krauncher_magic
```

```python
%%krauncher --pip torch
import torch
import torch.nn as nn

model = build_model().to("cuda")
losses = train(model, epochs, batch_size)   # epochs, batch_size taken from the notebook
accuracy = evaluate(model)
```

```
krauncher: auto --in epochs,batch_size --out model,losses,accuracy
krauncher: estimate — cv_training, ≥6 GB VRAM, 214.0 CU
Executing on worker-a1514da4: RTX 5060 Ti, 15GB, storage 2884 MB/s, net 800 Mbps
Epoch 1/3  avg_loss=2.3409          ← streamed live, with a GPU metric line
krauncher: done on RTX 5060 Ti in 131.2s — 3.42 KU → losses, accuracy
```

`losses` and `accuracy` are regular variables in the notebook afterwards,
produced on a GPU picked for the lowest price for this specific workload.
No flags are required: the transfer set is detected from the cell's code.

## Why per-cell

Each marked cell is one ephemeral task: no GPU session, no idle time billed,
no instance to choose. The broker classifies the cell's code (VRAM, workload
class, compute units), quotes it, and dispatches it to the cheapest GPU that
fits — across providers. Unmarked cells cost nothing.

No GPU state persists between cells (load-model-in-one-cell,
use-in-the-next does not work — group related work into a single cell).
Consecutive cells with the same requirements do reuse a warm worker when one
is still around (session affinity), and an opt-in idle hold in your account
settings keeps it around longer.

## What the magic does for you

- **Auto namespace** — free variables of the cell that exist in the notebook
  are sent as inputs; assigned names come back as outputs. Values are
  JSON-safe data; models/tensors stay on the worker and are reported, not
  fatal. Credential-shaped values and values over 1 MB are held back unless
  passed explicitly with `--in`.
- **Quote before run** — the analysis phase prices the cell (VRAM, compute
  units, IO) before anything is submitted; `--estimate` stops there.
- **Live feedback** — remote stdout/stderr stream into the cell output as
  they happen, plus a GPU utilisation/VRAM/elapsed progress line.
- **Hugging Face / S3 pre-fetch** — literal `load_dataset("org/name")`,
  `from_pretrained("org/name")` and `"s3://bucket/key"` references are
  downloaded by the data bridge *before* the container starts (the IO is
  metered as download, not compute) and served to your unmodified code.
  Credentials come from your account defaults (Storage → Credentials).
- **Non-blocking cells** — `--async` submits and returns a handle for later
  cells (`await kr_task` / `kr_task.result()`).
- **Restart the kernel = cancel** — in-flight tasks are cancelled and their
  holds released when the kernel exits.

## Options

| Flag | Meaning |
|---|---|
| `--in NAMES` | override the auto-detected inputs (comma-separated, repeatable) |
| `--out NAMES` | override the auto-detected outputs |
| `--pip PKGS` | pip packages installed in the sandbox before execution |
| `--vram N` | minimum GPU VRAM in GB (default: auto-classified from the code) |
| `--gpu-name S` | pin a GPU model (case-insensitive substring, e.g. `A4000`) |
| `--timeout N` | execution timeout in seconds (default 600) |
| `--dataset-size MB` | declared input size for the quote (e.g. private S3 objects) |
| `--async [NAME]` | non-blocking: inject a task handle as NAME (default `kr_task`) |
| `--estimate` | classify and print the quote only — do not run |

## Install & credentials

```bash
pip install krauncher-jupyter          # pulls the krauncher SDK
```

`CAS_API_KEY` (and optionally `CAS_BROKER_URL`) as environment variables or
in a `.env` next to the notebook — same as the SDK. Get a key at
[krauncher.com](https://krauncher.com) → Account → API Keys.

## Limitations

- Transferred values are JSON-safe data only (numbers, strings, lists,
  dicts) within a 16 MB inline budget; larger or binary data goes through
  data sources / volumes / the HF-S3 pre-fetch above.
- No GPU state across cells (see "Why per-cell").
- Dynamic Hub/S3 references (f-strings, variables) cannot be pre-fetched —
  they download inside execution and are billed as compute (a warning says
  so).
- Top-level `global` in a cell is not supported (the body runs as a
  function).

## Development

```bash
PYTHONPATH=.:../cas-client python -m pytest tests/
```

## License

MIT © 2026 Ilya Sergeev
