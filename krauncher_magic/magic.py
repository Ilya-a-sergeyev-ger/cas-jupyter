# Copyright (c) 2026 Ilya Sergeev. Licensed under the MIT License.

"""The %%krauncher cell magic — submit a cell as an ephemeral CaS task."""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import threading
import uuid
from typing import Any

from IPython.core import magic_arguments
from IPython.core.magic import Magics, cell_magic, magics_class

# Auto-transfer guards: values above the size limit and credential-shaped
# values are not sent automatically — an explicit --in overrides both.
_AUTO_INPUT_LIMIT_BYTES = 1024 * 1024
_SECRET_NAME_RE = re.compile(
    r"(?i)(^|_)(secrets?|tokens?|passwords?|passwd|pwd|api_?keys?"
    r"|credentials?|private_key|access_key|auth)(_|$)"
)
_SECRET_VALUE_RE = re.compile(
    r"^(sk-[A-Za-z0-9_-]{8,}|ghp_[A-Za-z0-9]{8,}|gho_[A-Za-z0-9]{8,}"
    r"|github_pat_|xox[baprs]-|AKIA[0-9A-Z]{16}|eyJ[A-Za-z0-9_-]{8,}\.)"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY"
)


# Session affinity: one id per kernel process — Restart the kernel starts a
# new session (and its in-flight sweep cancelled the old session's tasks).
_SESSION_ID = uuid.uuid4().hex[:8]

# Persistent background loop escorting --async tasks. E2E payload delivery
# and result collection live inside TaskHandle.wait(), so someone must keep
# awaiting the task after the submitting cell returns.
_ASYNC_LOOP: asyncio.AbstractEventLoop | None = None


def _escort_loop() -> asyncio.AbstractEventLoop:
    global _ASYNC_LOOP
    if _ASYNC_LOOP is None or _ASYNC_LOOP.is_closed():
        loop = asyncio.new_event_loop()
        threading.Thread(
            target=loop.run_forever, name="krauncher-escort", daemon=True,
        ).start()
        _ASYNC_LOOP = loop
    return _ASYNC_LOOP


class AsyncTask:
    """Handle injected by ``%%krauncher --async [NAME]`` for later cells.

    ``task.done()`` polls; ``task.result()`` blocks, injects the outputs
    into the notebook namespace and returns them; ``await task`` does the
    same on the kernel loop. Failures raise with the remote status/traceback.
    """

    def __init__(self, future, user_ns: dict, box: dict):
        self._future = future
        self._user_ns = user_ns
        self._box = box  # escort fills task_id after submission

    @property
    def task_id(self) -> str | None:
        return self._box.get("task_id")

    def done(self) -> bool:
        return self._future.done()

    def result(self, timeout: float | None = None) -> dict:
        values = self._future.result(timeout)
        self._user_ns.update(values)
        return values

    def __await__(self):
        async def _get():
            values = await asyncio.wrap_future(self._future)
            self._user_ns.update(values)
            return values
        return _get().__await__()

    def __repr__(self) -> str:
        if not self._future.done():
            state = "running"
        elif self._future.exception() is not None:
            state = f"failed: {self._future.exception()}"
        else:
            state = "done"
        return f"<krauncher async task {self.task_id or '?'} — {state}>"


def _session_group_id(min_vram_gb: int, gpu_name: str | None = None) -> str:
    """Affinity key for a cell: session + requirement class.

    Tier-1 group affinity pins to the group's worker WITHOUT re-checking
    requirements, so the key must never mix VRAM classes / GPU pins — cells
    with equal requirements share a warm worker, heavier ones re-route.
    """
    gid = f"nb-{_SESSION_ID}-v{min_vram_gb}"
    if gpu_name:
        gid += "-" + re.sub(r"[^a-z0-9]+", "", gpu_name.lower())
    return gid


def _auto_inputs(free: list[str], user_ns: dict) -> tuple[list[str], list[str]]:
    """Filter free-name candidates against the notebook namespace.

    Returns ``(names, notes)`` — names to send and human-readable notes about
    candidates that were held back (and how to send them anyway).
    """
    names: list[str] = []
    notes: list[str] = []
    unsafe: list[str] = []
    for n in free:
        if n not in user_ns:
            continue  # genuinely undefined — the remote NameError says it best
        v = user_ns[n]
        if inspect.ismodule(v):
            notes.append(f"{n}: module — import it inside the cell")
            continue
        if callable(v):
            unsafe.append(n)  # notebook-defined function/class: not transferable
            continue
        if _SECRET_NAME_RE.search(n) or (
            isinstance(v, str) and _SECRET_VALUE_RE.match(v)
        ):
            notes.append(f"{n}: looks like a credential — not sent (pass --in {n} to send)")
            continue
        try:
            size = len(json.dumps(v).encode("utf-8"))
        except (TypeError, ValueError):
            unsafe.append(n)
            continue
        if size > _AUTO_INPUT_LIMIT_BYTES:
            notes.append(
                f"{n}: {size / (1024 * 1024):.1f} MB — over the 1 MB auto limit "
                f"(pass --in {n} to send)"
            )
            continue
        names.append(n)
    if unsafe:
        notes.append(
            "not auto-sent (non-transferable): " + ", ".join(unsafe)
            + " — recreate inside the cell or use a volume / data source"
        )
    return names, notes


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
                              help="comma-separated notebook variables sent to the task "
                                   "(default: auto-detect from the cell's free variables)")
    @magic_arguments.argument("--out", dest="outputs", action="append", metavar="NAMES",
                              help="comma-separated variables returned into the notebook "
                                   "(default: auto-detect from the cell's assignments)")
    @magic_arguments.argument("--pip", action="append", metavar="PKGS",
                              help="comma-separated pip packages for the sandbox")
    @magic_arguments.argument("--vram", type=int, default=None,
                              help="minimum GPU VRAM in GB (default: auto-classify)")
    @magic_arguments.argument("--timeout", type=int, default=600,
                              help="execution timeout in seconds (default 600)")
    @magic_arguments.argument("--gpu-name", default=None,
                              help="pin a GPU model (substring, e.g. 'A4000')")
    @magic_arguments.argument("--dataset-size", dest="dataset_size", type=float, default=None,
                              help="declared input data size in MB for the quote "
                                   "(e.g. private S3 objects the client cannot size)")
    @magic_arguments.argument("--async", dest="async_name", nargs="?", const="kr_task",
                              default=None, metavar="NAME",
                              help="non-blocking: submit and return immediately; a handle "
                                   "is injected as NAME (default kr_task) for later cells")
    @magic_arguments.argument("--estimate", action="store_true",
                              help="classify and quote only — do not run")
    @cell_magic
    def krauncher(self, line: str, cell: str) -> None:
        args = magic_arguments.parse_argstring(self.krauncher, line)
        pip = _split_names(args.pip)

        # --in / --out are exact overrides; when omitted, detect the transfer
        # set from the cell's AST (free variables in, assigned names out).
        auto_in = args.inputs is None
        auto_out = args.outputs is None
        if auto_in or auto_out:
            from krauncher.codeblock import analyze_names
            try:
                free, assigned = analyze_names(cell)
            except Exception:
                free, assigned = [], []  # submission reports the syntax error

        if auto_in:
            inputs, notes = _auto_inputs(free, self.shell.user_ns)
            for note in notes:
                print(f"krauncher: {note}")
        else:
            inputs = _split_names(args.inputs)
            missing = [n for n in inputs if n not in self.shell.user_ns]
            if missing:
                print(f"krauncher: --in {', '.join(missing)}: not defined in the notebook")
                return
        outputs = assigned if auto_out else _split_names(args.outputs)
        if auto_in or auto_out:
            print(f"krauncher: auto --in {','.join(inputs) or '(none)'} "
                  f"--out {','.join(outputs) or '(none)'}")

        call_values = {n: self.shell.user_ns[n] for n in inputs}

        # HF references: translate literal load_dataset/from_pretrained calls
        # into pre-fetched data-bridge downloads (hub-cache layout) — the IO
        # runs before the container starts and lands in the download phase.
        from krauncher.hf import CACHE_FRAGMENT, detect_hf_refs
        hf_urls, hf_dynamic = detect_hf_refs(cell)
        for d in hf_dynamic:
            print(f"krauncher: {d}: dynamic HF reference — downloads in-code, "
                  f"IO will be billed as compute")

        # S3 references: exact-object literals are pre-fetched by the bridge
        # and rewritten to the local mount path (no cache layer to rely on);
        # credentials come from the user defaults at dispatch.
        from krauncher.s3 import detect_s3_refs, rewrite_s3_refs, s3_local_mapping
        s3_urls, s3_notes = detect_s3_refs(cell)
        s3_map, s3_map_notes = s3_local_mapping(s3_urls) if s3_urls else ({}, [])
        for n in (*s3_notes, *s3_map_notes):
            print(f"krauncher: {n}")
        if s3_map:
            cell = rewrite_s3_refs(cell, s3_map)
            for u, p in s3_map.items():
                print(f"krauncher: s3 pre-fetch: {u} → {p}")

        data_urls = [u + CACHE_FRAGMENT for u in hf_urls] + list(s3_map) or None

        # Fresh client per cell: each execution runs on its own private event
        # loop (see _run_sync), and the cell code changes between runs anyway,
        # so there is no cross-cell client state worth keeping.
        from krauncher import KrauncherClient, KrauncherError
        from krauncher.values import decode_outputs

        # --estimate stops after the analysis phase — no estimate_only stubs.
        client = KrauncherClient()

        async def _phase1():
            # Size the pre-fetch first: it feeds cu_io / disk in the quote.
            # Explicit --dataset-size wins (private S3 objects cannot be
            # sized client-side); otherwise the HF Hub API sizes hf refs.
            dataset_mb = args.dataset_size
            if hf_urls:
                if dataset_mb is None:
                    from krauncher.hf import hf_size_mb
                    dataset_mb = await hf_size_mb(hf_urls)
                size_str = f" ({dataset_mb:.0f} MB)" if dataset_mb else ""
                print("krauncher: hf pre-fetch: "
                      + ", ".join(u.removeprefix("hf://") for u in hf_urls)
                      + size_str)
            # Analysis request: quote before anything is submitted.
            quote = await client.estimate_code(
                cell,
                inputs=call_values,
                outputs=outputs,
                lenient_outputs=auto_out,
                vram_gb=args.vram,
                dataset_size=dataset_mb,
            )
            return dataset_mb, quote

        def _run_kwargs(quote, dataset_mb, stream):
            # Phase 2 kwargs — precomputed classification (no re-analysis)
            # + session affinity keyed by the requirement class.
            return dict(
                inputs=call_values,
                outputs=outputs,
                # Auto-detected outputs may be unset or non-JSON-safe — drop
                # them remotely instead of failing the task.
                lenient_outputs=auto_out,
                classification=quote,
                group_id=_session_group_id(quote.min_vram_gb, args.gpu_name),
                gpu_name=args.gpu_name,
                pip=pip or None,
                timeout=args.timeout,
                data_urls=data_urls,
                dataset_size=dataset_mb,
                stream_stderr=stream,
            )

        if args.async_name:
            # Non-blocking: quote synchronously, then hand the submission and
            # the whole escort (payload delivery + wait + decode) to the
            # background loop. No console mirroring — the cell is long gone.
            try:
                dataset_mb, quote = _run_sync(_phase1())
            except KrauncherError as exc:
                print(f"krauncher: {exc}")
                return
            self._print_quote(quote)
            if args.estimate:
                return
            box: dict = {}

            async def _escort():
                handle = await client.run_code(
                    cell, **_run_kwargs(quote, dataset_mb, stream=False),
                )
                box["task_id"] = handle.task_id
                result = await handle.wait(timeout=args.timeout + 600)
                if result.status != "completed":
                    raise KrauncherError(
                        f"task {result.status}"
                        + (f"\n{result.traceback}" if result.traceback else "")
                    )
                returned = outputs
                if auto_out and isinstance(result.output, dict):
                    returned = [n for n in outputs if n in result.output]
                return decode_outputs(result.output, returned)

            future = asyncio.run_coroutine_threadsafe(_escort(), _escort_loop())
            task = AsyncTask(future, self.shell.user_ns, box)
            self.shell.user_ns[args.async_name] = task
            print(f"krauncher: async task → {args.async_name}  "
                  f"(poll .done(), values via .result() or `await {args.async_name}`)")
            return

        async def _submit():
            dataset_mb, quote = await _phase1()
            self._print_quote(quote)
            if args.estimate:
                return None
            # Live feedback: wait() mirrors remote stdout/stderr and the GPU
            # metric progress line into the cell output.
            handle = await client.run_code(
                cell, **_run_kwargs(quote, dataset_mb, stream=True),
            )
            # Client-side wait must outlast the remote execution timeout, or a
            # long task is abandoned before it finishes (the --async path does
            # the same). --timeout governs both remote kill and client wait.
            return await handle.wait(timeout=args.timeout + 600)

        try:
            result = _run_sync(_submit())
        except KrauncherError as exc:
            print(f"krauncher: {exc}")
            return
        if args.estimate or result is None:
            return

        # Remote stdout was already mirrored live during wait() — no final echo.
        if result.status != "completed":
            print(f"krauncher: task {result.status}"
                  + (f"\n{result.traceback}" if result.traceback else ""))
            return

        returned = outputs
        if auto_out and isinstance(result.output, dict):
            returned = [n for n in outputs if n in result.output]
            dropped = [n for n in outputs if n not in result.output]
            if dropped:
                print("krauncher: not returned (non-JSON-safe or unset): "
                      + ", ".join(dropped))
        try:
            values = decode_outputs(result.output, returned)
        except KrauncherError as exc:
            print(f"krauncher: {exc}")
            return
        self.shell.user_ns.update(values)

        cost = (f"{result.total_charged_ku:.2f} KU"
                if result.total_charged_ku else "n/a")
        print(f"krauncher: done on {result.actual_gpu} in "
              f"{result.execution_time_sec:.1f}s — {cost}"
              + (f" → {', '.join(returned)}" if returned else ""))

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
