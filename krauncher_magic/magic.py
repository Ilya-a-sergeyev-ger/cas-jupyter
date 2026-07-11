# Copyright (c) 2026 Ilya Sergeev. Licensed under the MIT License.

"""The %%krauncher cell magic — submit a cell as an ephemeral CaS task."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from IPython.core import magic_arguments
from IPython.core.magic import Magics, cell_magic, magics_class


def _run_sync(coro) -> Any:
    """Run a coroutine to completion from the (sync) magic context.

    ipykernel's own event loop is busy running the cell, so the task client
    gets a private loop in a worker thread. Jupyter's Interrupt (■) raises
    KeyboardInterrupt in the main thread only — propagate it as cancellation
    of the coroutine so the client's normal Ctrl-C path runs (cancel-on-
    abandon: broker DELETE + relay CancelTask stop the task and release the
    hold).
    """
    box: dict[str, Any] = {}
    done = threading.Event()

    def _target() -> None:
        loop = asyncio.new_event_loop()
        task = loop.create_task(coro)

        def _cancel() -> None:
            try:
                loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:
                pass  # loop already closed — nothing left to cancel

        box["cancel"] = _cancel
        try:
            box["value"] = loop.run_until_complete(task)
        except BaseException as exc:  # noqa: BLE001 — surfaced below
            box["error"] = exc
        finally:
            loop.close()
            done.set()

    thread = threading.Thread(target=_target, name="krauncher-cell", daemon=True)
    thread.start()
    try:
        # Completion is signalled via an Event, NOT Thread.join():
        # a KeyboardInterrupt landing inside join(timeout) corrupts the
        # Thread state (is_alive goes False while the thread still runs).
        # Bounded waits keep interrupt delivery prompt on all platforms.
        while not done.wait(0.2):
            pass
    except KeyboardInterrupt:
        cancel = box.get("cancel")
        if cancel:
            cancel()
        done.wait(timeout=30)  # let cancel-on-abandon inside wait() finish
        raise
    if "error" in box:
        raise box["error"]
    return box["value"]


def _split_names(chunks: list[str] | None) -> list[str]:
    """['a,b', 'c'] -> ['a', 'b', 'c'] (repeatable comma-separated flags)."""
    names: list[str] = []
    for chunk in chunks or []:
        names.extend(n.strip() for n in chunk.split(",") if n.strip())
    return names


@magics_class
class KrauncherMagics(Magics):
    """%%krauncher — run the cell on the cheapest suitable remote GPU."""

    @magic_arguments.magic_arguments()
    @magic_arguments.argument("--in", dest="inputs", action="append", metavar="NAMES",
                              help="comma-separated notebook variables sent to the task")
    @magic_arguments.argument("--out", dest="outputs", action="append", metavar="NAMES",
                              help="comma-separated variables returned into the notebook")
    @magic_arguments.argument("--pip", action="append", metavar="PKGS",
                              help="comma-separated pip packages for the sandbox")
    @magic_arguments.argument("--vram", type=int, default=None,
                              help="minimum GPU VRAM in GB (default: auto-classify)")
    @magic_arguments.argument("--timeout", type=int, default=600,
                              help="execution timeout in seconds (default 600)")
    @magic_arguments.argument("--gpu-name", default=None,
                              help="pin a GPU model (substring, e.g. 'A4000')")
    @magic_arguments.argument("--estimate", action="store_true",
                              help="classify and quote only — do not run")
    @cell_magic
    def krauncher(self, line: str, cell: str) -> None:
        args = magic_arguments.parse_argstring(self.krauncher, line)
        inputs = _split_names(args.inputs)
        outputs = _split_names(args.outputs)
        pip = _split_names(args.pip)

        missing = [n for n in inputs if n not in self.shell.user_ns]
        if missing:
            print(f"krauncher: --in {', '.join(missing)}: not defined in the notebook")
            return
        call_values = {n: self.shell.user_ns[n] for n in inputs}

        # Fresh client per cell: each execution runs on its own private event
        # loop (see _run_sync), and the cell code changes between runs anyway,
        # so there is no cross-cell client state worth keeping.
        from krauncher import KrauncherClient, KrauncherError
        from krauncher.values import decode_outputs

        client = KrauncherClient(estimate_only=args.estimate or None)

        async def _submit():
            handle = await client.run_code(
                cell,
                inputs=call_values,
                outputs=outputs,
                vram_gb=args.vram,
                gpu_name=args.gpu_name,
                pip=pip or None,
                timeout=args.timeout,
            )
            quote = getattr(handle, "classification", None)
            if quote is not None:
                self._print_quote(quote)
            if args.estimate:
                return None
            return await handle

        try:
            result = _run_sync(_submit())
        except KrauncherError as exc:
            print(f"krauncher: {exc}")
            return
        if args.estimate or result is None:
            return

        if result.stdout:
            print(result.stdout, end="")
        if result.status != "completed":
            print(f"krauncher: task {result.status}"
                  + (f"\n{result.traceback}" if result.traceback else ""))
            return

        try:
            values = decode_outputs(result.output, outputs)
        except KrauncherError as exc:
            print(f"krauncher: {exc}")
            return
        self.shell.user_ns.update(values)

        cost = (f"{result.total_charged_ku:.2f} KU"
                if result.total_charged_ku else "n/a")
        print(f"krauncher: done on {result.actual_gpu} in "
              f"{result.execution_time_sec:.1f}s — {cost}"
              + (f" → {', '.join(outputs)}" if outputs else ""))

    @staticmethod
    def _print_quote(c: Any) -> None:
        """Pre-run assay line from the analyzer classification."""
        parts = []
        if getattr(c, "workload_type", None):
            parts.append(c.workload_type)
        if getattr(c, "min_vram_gb", None):
            parts.append(f"≥{c.min_vram_gb} GB VRAM")
        if getattr(c, "compute_units", None):
            parts.append(f"{c.compute_units:.1f} CU")
        if getattr(c, "predicted_sec", None):
            parts.append(f"~{c.predicted_sec:.0f}s (ref)")
        if parts:
            print("krauncher: estimate — " + ", ".join(parts))
