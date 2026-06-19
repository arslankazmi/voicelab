"""Tests for the heavy-model memory manager (OOM safety). No real models loaded."""

from __future__ import annotations

from voicelab.tts import _model_manager as mm


def test_resident_model_reused_not_reloaded():
    calls = {"n": 0}

    def loader():
        calls["n"] += 1
        return f"model-{calls['n']}"

    with mm.heavy_session("a", loader) as m1:
        assert m1 == "model-1"
    with mm.heavy_session("a", loader) as m2:
        assert m2 == "model-1"  # reused
    assert calls["n"] == 1
    assert mm.resident_model_name() == "a"


def test_switching_engines_evicts_previous():
    loaded: list[str] = []

    def mk(name):
        def _loader():
            loaded.append(name)
            return f"obj-{name}"

        return _loader

    with mm.heavy_session("a", mk("a")) as m:
        assert m == "obj-a"
    assert mm.resident_model_name() == "a"

    with mm.heavy_session("b", mk("b")) as m:  # different engine → evicts "a"
        assert m == "obj-b"
    assert mm.resident_model_name() == "b"

    with mm.heavy_session("a", mk("a")) as m:  # "a" was evicted → reloads
        assert m == "obj-a"
    assert loaded == ["a", "b", "a"]  # single residency: each switch reloads


def test_preflight_refuses_when_memory_low(monkeypatch):
    monkeypatch.setattr(mm, "_free_mb", lambda: 100.0)  # below the 3000 MB floor
    called = {"n": 0}

    def loader():
        called["n"] += 1
        return "model"

    with mm.heavy_session("x", loader) as m:
        assert m is None  # refused — never loads → host can't OOM
    assert called["n"] == 0
    assert mm.resident_model_name() is None


def test_loader_failure_yields_none_and_evicts(monkeypatch):
    monkeypatch.setattr(mm, "_free_mb", lambda: 999_999.0)

    def boom():
        raise RuntimeError("load failed")

    with mm.heavy_session("y", boom) as m:
        assert m is None
    assert mm.resident_model_name() is None  # cleaned up
