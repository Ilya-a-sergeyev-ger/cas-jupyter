# Copyright (c) 2026 Ilya Sergeev. Licensed under the MIT License.

"""Adapter-level tests — UI-syntax parsing only; transfer/codegen logic
lives in krauncher (cas-client) and is tested there."""

import _thread
import asyncio
import threading

import pytest

from krauncher_magic.magic import _run_sync, _split_names


def test_split_names_repeatable_comma_flags():
    assert _split_names(["a,b", "c"]) == ["a", "b", "c"]


def test_split_names_strips_and_skips_empty():
    assert _split_names([" a , ", ",b"]) == ["a", "b"]


def test_split_names_none():
    assert _split_names(None) == []


def test_run_sync_returns_value():
    async def coro():
        return 42

    assert _run_sync(coro()) == 42


def test_interrupt_cancels_coroutine():
    """Jupyter's Interrupt (KeyboardInterrupt in the main thread) must reach
    the coroutine as CancelledError so the client's cancel-on-abandon path
    (TaskHandle.wait) runs exactly as on a terminal Ctrl-C."""
    state = {}

    async def coro():
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            state["cancelled"] = True  # what wait() sees before cancel-on-abandon
            raise

    # Emulate ipykernel's interrupt: KeyboardInterrupt lands in the main
    # thread while it blocks in thread.join().
    threading.Timer(0.2, _thread.interrupt_main).start()
    with pytest.raises(KeyboardInterrupt):
        _run_sync(coro())
    assert state.get("cancelled") is True
