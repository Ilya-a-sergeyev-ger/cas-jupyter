# Copyright (c) 2026 Ilya Sergeev. Licensed under the MIT License.

"""Adapter-level tests — UI-syntax parsing only; transfer/codegen logic
lives in krauncher (cas-client) and is tested there."""

from krauncher_magic.magic import _split_names


def test_split_names_repeatable_comma_flags():
    assert _split_names(["a,b", "c"]) == ["a", "b", "c"]


def test_split_names_strips_and_skips_empty():
    assert _split_names([" a , ", ",b"]) == ["a", "b"]


def test_split_names_none():
    assert _split_names(None) == []
