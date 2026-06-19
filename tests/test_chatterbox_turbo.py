"""Tests for ChatterboxTurboTts backend and its registry entry.

CI-safe: all tests mock chatterbox.tts_turbo — the real model is never
loaded.  Real-synthesis tests are guarded by @pytest.mark.skipif.
"""

from __future__ import annotations

import importlib
import importlib.util
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ===========================================================================
# ChatterboxTurboTts — mocked (never loads torch / real model)
# ===========================================================================


class TestChatterboxTurboTtsMocked:
    def _make_fake_audio(self, length: int = 22050) -> MagicMock:
        """Return a MagicMock tensor whose .squeeze().detach().cpu().numpy() is zeros."""
        fake_audio = MagicMock()
        fake_audio.squeeze.return_value.detach.return_value.cpu.return_value.numpy.return_value = (
            np.zeros(length, dtype=np.float32)
        )
        return fake_audio

    def test_synthesize_returns_none_when_not_installed(self):
        """If chatterbox.tts_turbo is not importable, synthesize returns None."""
        from voicelab.tts import chatterbox_turbo as turbo_mod

        turbo_mod._model = None
        turbo_mod._model_load_attempted = False

        with patch.dict("sys.modules", {"chatterbox": None, "chatterbox.tts_turbo": None}):
            turbo_mod._model_load_attempted = False
            from voicelab.tts.chatterbox_turbo import ChatterboxTurboTts

            tts = ChatterboxTurboTts()
            result = tts.synthesize("hello", "default", {})

        assert result is None

    def test_list_voices_returns_two_entries(self):
        from voicelab.tts.chatterbox_turbo import ChatterboxTurboTts

        tts = ChatterboxTurboTts()
        voices = tts.list_voices()
        assert len(voices) == 2
        ids = {v["id"] for v in voices}
        assert "default" in ids
        assert "cloned" in ids
        assert all(v["category"] == "chatterbox-turbo" for v in voices)

    def test_synthesize_maps_exaggeration_to_temperature(self):
        """exaggeration is forwarded AND also mapped to temperature."""
        from voicelab.tts import chatterbox_turbo as turbo_mod
        from voicelab.tts._chatter_common import exaggeration_to_temperature

        turbo_mod._model = None
        turbo_mod._model_load_attempted = False

        fake_audio = self._make_fake_audio()
        fake_model = MagicMock()
        fake_model.sr = 22050
        fake_model.generate.return_value = fake_audio

        turbo_mod._model = fake_model
        turbo_mod._model_load_attempted = True

        from voicelab.tts.chatterbox_turbo import ChatterboxTurboTts

        exag = 0.8
        tts = ChatterboxTurboTts()
        result = tts.synthesize("test text", "default", {"exaggeration": exag, "cfg_weight": 0.3})

        assert result is not None
        assert result[:4] == b"RIFF"

        call_kwargs = fake_model.generate.call_args
        assert call_kwargs is not None
        _, kw = call_kwargs
        expected_temp = exaggeration_to_temperature(exag)
        assert abs(kw.get("temperature", 0) - expected_temp) < 1e-9

    def test_synthesize_temperature_equals_exaggeration_to_temperature(self):
        """temperature == exaggeration_to_temperature(exaggeration) for all test values."""
        from voicelab.tts import _model_manager as _mm
        from voicelab.tts import chatterbox_turbo as turbo_mod
        from voicelab.tts._chatter_common import exaggeration_to_temperature

        for exag in [0.0, 0.25, 0.5, 0.75, 1.0]:
            _mm.evict()  # drop the previous resident fake so this iteration's fake loads
            turbo_mod._model = None
            turbo_mod._model_load_attempted = False

            fake_audio = self._make_fake_audio()
            fake_model = MagicMock()
            fake_model.sr = 22050
            fake_model.generate.return_value = fake_audio

            turbo_mod._model = fake_model
            turbo_mod._model_load_attempted = True

            from voicelab.tts.chatterbox_turbo import ChatterboxTurboTts

            tts = ChatterboxTurboTts()
            tts.synthesize("hello", "default", {"exaggeration": exag})

            _, kw = fake_model.generate.call_args
            assert abs(kw.get("temperature", 0) - exaggeration_to_temperature(exag)) < 1e-9, (
                f"exag={exag}: expected temp={exaggeration_to_temperature(exag)}, got {kw.get('temperature')}"
            )

    def test_synthesize_passes_reference_audio_when_given(self):
        """audio_prompt_path is forwarded when reference_audio is in settings."""
        from voicelab.tts import chatterbox_turbo as turbo_mod

        turbo_mod._model = None
        turbo_mod._model_load_attempted = False

        fake_audio = self._make_fake_audio()
        fake_model = MagicMock()
        fake_model.sr = 22050
        fake_model.generate.return_value = fake_audio

        turbo_mod._model = fake_model
        turbo_mod._model_load_attempted = True

        from voicelab.tts.chatterbox_turbo import ChatterboxTurboTts

        tts = ChatterboxTurboTts()
        result = tts.synthesize(
            "clone me",
            "cloned",
            {"reference_audio": "/tmp/ref.wav", "exaggeration": 0.5},
        )

        assert result is not None
        assert result[:4] == b"RIFF"

        _, kw = fake_model.generate.call_args
        assert kw.get("audio_prompt_path") == "/tmp/ref.wav"

    def test_synthesize_graceful_none_on_generate_exception(self):
        """If model.generate() raises, synthesize returns None (never raises)."""
        from voicelab.tts import chatterbox_turbo as turbo_mod

        fake_model = MagicMock()
        fake_model.generate.side_effect = RuntimeError("turbo boom")

        turbo_mod._model = fake_model
        turbo_mod._model_load_attempted = True

        from voicelab.tts.chatterbox_turbo import ChatterboxTurboTts

        tts = ChatterboxTurboTts()
        result = tts.synthesize("boom", "default", {})
        assert result is None

    def test_synthesize_returns_wav_bytes_with_mocked_model(self):
        """Full happy-path: returns RIFF WAV bytes."""
        from voicelab.tts import chatterbox_turbo as turbo_mod

        turbo_mod._model = None
        turbo_mod._model_load_attempted = False

        fake_audio = self._make_fake_audio(44100)
        fake_model = MagicMock()
        fake_model.sr = 24000
        fake_model.generate.return_value = fake_audio

        turbo_mod._model = fake_model
        turbo_mod._model_load_attempted = True

        from voicelab.tts.chatterbox_turbo import ChatterboxTurboTts

        tts = ChatterboxTurboTts()
        result = tts.synthesize("Hello from Turbo!", "default", {"exaggeration": 0.5})

        assert result is not None
        assert isinstance(result, bytes)
        assert result[:4] == b"RIFF"
        assert result[8:12] == b"WAVE"

    def test_load_model_graceful_import_error(self):
        """_load_model returns None without raising when import fails."""
        from voicelab.tts import chatterbox_turbo as turbo_mod

        turbo_mod._model = None
        turbo_mod._model_load_attempted = False

        with patch.dict("sys.modules", {"chatterbox.tts_turbo": None}):
            turbo_mod._model_load_attempted = False
            result = turbo_mod._load_model()

        assert result is None


# ===========================================================================
# Registry tests for chatterbox-turbo
# ===========================================================================


class TestChatterboxTurboRegistry:
    def test_turbo_in_registry(self):
        """chatterbox-turbo appears in the registry."""
        from voicelab.tts.registry import _REGISTRY

        entry = next((b for b in _REGISTRY if b.name == "chatterbox-turbo"), None)
        assert entry is not None

    def test_turbo_license_is_mit(self):
        from voicelab.tts.registry import _REGISTRY

        entry = next(b for b in _REGISTRY if b.name == "chatterbox-turbo")
        assert entry.license == "MIT"

    def test_turbo_requires_key_false(self):
        from voicelab.tts.registry import _REGISTRY

        entry = next(b for b in _REGISTRY if b.name == "chatterbox-turbo")
        assert entry.requires_key is False

    def test_turbo_has_turbo_tag(self):
        from voicelab.tts.registry import _REGISTRY

        entry = next(b for b in _REGISTRY if b.name == "chatterbox-turbo")
        assert "turbo" in entry.tags

    def test_turbo_import_check_is_tts_turbo_module(self):
        from voicelab.tts.registry import _REGISTRY

        entry = next(b for b in _REGISTRY if b.name == "chatterbox-turbo")
        assert entry.import_check == "chatterbox.tts_turbo"

    def test_turbo_higher_priority_number_than_local(self):
        """chatterbox-turbo is opt-in, so it has a higher priority number (lower preference) than local."""
        from voicelab.tts.registry import _REGISTRY

        local_p = next(b.priority for b in _REGISTRY if b.name == "local")
        turbo_p = next(b.priority for b in _REGISTRY if b.name == "chatterbox-turbo")
        assert turbo_p > local_p

    def test_registry_priorities_still_sorted(self):
        """Adding chatterbox-turbo must not break the sorted-priority invariant."""
        from voicelab.tts.registry import _REGISTRY

        priorities = [b.priority for b in _REGISTRY]
        assert priorities == sorted(priorities), (
            f"Registry must be sorted by priority, got: {priorities}"
        )

    def test_turbo_listed_in_list_backends(self):
        from voicelab.tts.registry import list_backends

        names = {b["name"] for b in list_backends()}
        assert "chatterbox-turbo" in names

    def test_turbo_unavailable_when_import_check_missing(self):
        """When chatterbox.tts_turbo can't be imported, is_available returns False."""
        import voicelab.tts.registry as registry_mod

        turbo_info = next(b for b in registry_mod._REGISTRY if b.name == "chatterbox-turbo")

        with patch("importlib.import_module", side_effect=ImportError("no tts_turbo")):
            assert registry_mod.is_available(turbo_info) is False

    def test_get_tts_for_turbo_resolves_with_mocked_module(self):
        """engine='chatterbox-turbo' resolves to ChatterboxTurboTts when mocked available."""
        import voicelab.tts.registry as registry_mod
        from voicelab.tts.chatterbox_turbo import ChatterboxTurboTts

        fake_mod = MagicMock()
        mocks = {
            "chatterbox": fake_mod,
            "chatterbox.tts_turbo": fake_mod,
        }
        with patch.dict("sys.modules", mocks):
            with patch.object(registry_mod, "_instantiate") as mock_inst:
                mock_inst.side_effect = lambda info, **kw: (
                    ChatterboxTurboTts() if info.name == "chatterbox-turbo" else None
                )
                tts = registry_mod.get_tts_for_engine("chatterbox-turbo", api_key=None)

        assert isinstance(tts, ChatterboxTurboTts)

    def test_turbo_not_in_auto_chain_when_unavailable(self):
        """Auto mode never picks chatterbox-turbo when its module is not importable."""
        import voicelab.tts.registry as registry_mod

        original = registry_mod.is_available

        def mock_avail(info, *, api_key=None):
            if info.name == "chatterbox-turbo":
                return False
            return original(info, api_key=api_key)

        with patch.object(registry_mod, "is_available", side_effect=mock_avail):
            tts = registry_mod.get_tts_for_engine("auto", api_key=None)

        from voicelab.tts.chatterbox_turbo import ChatterboxTurboTts

        assert not isinstance(tts, ChatterboxTurboTts)


# ===========================================================================
# Real-synthesis test — SKIPPED unless chatterbox-tts is installed
# ===========================================================================


@pytest.mark.skipif(
    importlib.util.find_spec("chatterbox") is None,
    reason="chatterbox-tts not installed (uv sync --extra clone)",
)
class TestChatterboxTurboReal:
    def test_turbo_synthesize_short_sentence(self):
        """Real ChatterboxTurbo synthesis: returns non-empty WAV bytes."""
        from voicelab.tts import chatterbox_turbo as turbo_mod

        turbo_mod._model = None
        turbo_mod._model_load_attempted = False

        from voicelab.tts.chatterbox_turbo import ChatterboxTurboTts

        tts = ChatterboxTurboTts()
        result = tts.synthesize(
            "The quick brown fox jumps over the lazy dog.",
            "default",
            {"exaggeration": 0.5},
        )

        assert result is not None, "ChatterboxTurboTts returned None — check model download"
        assert len(result) > 1000, "Expected substantial WAV bytes"
        assert result[:4] == b"RIFF", f"Expected RIFF header, got {result[:4]!r}"
        assert result[8:12] == b"WAVE", "Expected WAVE format marker"

    def test_turbo_list_voices(self):
        from voicelab.tts.chatterbox_turbo import ChatterboxTurboTts

        tts = ChatterboxTurboTts()
        voices = tts.list_voices()
        assert len(voices) >= 2
        ids = {v["id"] for v in voices}
        assert "default" in ids
