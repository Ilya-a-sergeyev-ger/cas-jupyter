# Copyright (c) 2026 Ilya Sergeev. Licensed under the MIT License.

"""krauncher_magic — %%krauncher cell magic for Jupyter.

Runs a marked notebook cell as an ephemeral CaS task on the cheapest suitable
GPU and returns the declared outputs into the notebook namespace.

Usage::

    %load_ext krauncher_magic

    %%krauncher --in df,epochs --out model,accuracy --pip torch
    model = train(df, epochs)
    accuracy = evaluate(model, df)

Phase 0: explicit transfer only (--in / --out are required to move variables);
inputs/outputs cross the boundary via stdlib pickle — plain data (arrays,
DataFrames, scalars) transfers, functions/classes defined in the notebook do
not (cloudpickle is Phase 1). Credentials: CAS_API_KEY / CAS_BROKER_URL env
vars or .env, same as the krauncher SDK.
"""

__version__ = "0.1.0"


def load_ipython_extension(ipython):
    # Imported lazily: transfer/codegen are usable (and testable) without IPython.
    from .magic import KrauncherMagics

    ipython.register_magics(KrauncherMagics)
