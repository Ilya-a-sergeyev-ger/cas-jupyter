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


# ---------------------------------------------------------------------------
# auto namespace guards (_auto_inputs)
# ---------------------------------------------------------------------------

from krauncher_magic.magic import _auto_inputs


def test_auto_inputs_plain_values_pass():
    names, notes = _auto_inputs(["epochs", "cfg"], {"epochs": 3, "cfg": {"lr": 0.1}})
    assert names == ["epochs", "cfg"]
    assert notes == []


def test_auto_inputs_undefined_skipped_silently():
    names, notes = _auto_inputs(["nope"], {})
    assert names == [] and notes == []


def test_auto_inputs_module_noted():
    import json as mod
    names, notes = _auto_inputs(["mod"], {"mod": mod})
    assert names == []
    assert any("import it inside the cell" in n for n in notes)


def test_auto_inputs_callable_not_transferable():
    names, notes = _auto_inputs(["fn"], {"fn": lambda: 1})
    assert names == []
    assert any("non-transferable" in n for n in notes)


def test_auto_inputs_secret_name_held_back():
    names, notes = _auto_inputs(["api_key"], {"api_key": "abc"})
    assert names == []
    assert any("credential" in n and "--in api_key" in n for n in notes)


def test_auto_inputs_secret_value_held_back():
    names, notes = _auto_inputs(["s"], {"s": "sk-abcdefghijkl"})
    assert names == []
    assert any("credential" in n for n in notes)


def test_auto_inputs_size_guard():
    big = "x" * (2 * 1024 * 1024)
    names, notes = _auto_inputs(["blob"], {"blob": big})
    assert names == []
    assert any("1 MB auto limit" in n and "--in blob" in n for n in notes)


def test_auto_inputs_non_json_safe_noted():
    names, notes = _auto_inputs(["obj"], {"obj": object()})
    assert names == []
    assert any("non-transferable" in n for n in notes)


def test_auto_inputs_author_is_not_a_secret():
    names, notes = _auto_inputs(["author"], {"author": "ilya"})
    assert names == ["author"] and notes == []


# ---------------------------------------------------------------------------
# session affinity (_session_group_id)
# ---------------------------------------------------------------------------

from krauncher_magic import magic as magic_mod
from krauncher_magic.magic import _session_group_id


def test_session_group_id_stable_within_session():
    assert _session_group_id(6) == _session_group_id(6)
    assert magic_mod._SESSION_ID in _session_group_id(6)


def test_session_group_id_varies_by_vram_class():
    assert _session_group_id(6) != _session_group_id(30)
    assert _session_group_id(30).endswith("-v30")


def test_session_group_id_gpu_pin_in_key():
    plain = _session_group_id(24)
    pinned = _session_group_id(24, "RTX 4090")
    assert plain != pinned
    assert pinned.endswith("-rtx4090")


# ---------------------------------------------------------------------------
# --async escort (AsyncTask + background loop)
# ---------------------------------------------------------------------------

import asyncio as _asyncio

from krauncher_magic.magic import AsyncTask, _escort_loop


def test_escort_loop_singleton_and_alive():
    loop1 = _escort_loop()
    loop2 = _escort_loop()
    assert loop1 is loop2 and loop1.is_running()


def test_async_task_result_injects_values():
    ns = {}
    async def work():
        return {"a": 1, "b": 2}
    fut = _asyncio.run_coroutine_threadsafe(work(), _escort_loop())
    task = AsyncTask(fut, ns, {"task_id": "t-1"})
    assert task.result(timeout=5) == {"a": 1, "b": 2}
    assert ns == {"a": 1, "b": 2}
    assert task.done() and task.task_id == "t-1"


def test_async_task_failure_raises():
    async def boom():
        raise RuntimeError("remote failed")
    fut = _asyncio.run_coroutine_threadsafe(boom(), _escort_loop())
    task = AsyncTask(fut, {}, {})
    with pytest.raises(RuntimeError, match="remote failed"):
        task.result(timeout=5)
    assert "failed" in repr(task)


async def _await_it(task):
    return await task


def test_async_task_awaitable():
    ns = {}
    async def work():
        return {"x": 9}
    fut = _asyncio.run_coroutine_threadsafe(work(), _escort_loop())
    task = AsyncTask(fut, ns, {})
    values = _asyncio.run(_await_it(task))
    assert values == {"x": 9} and ns == {"x": 9}
