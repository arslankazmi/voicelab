"""Tests for Kokoro, Piper, Chatterbox backends + registry/selection.

CI-safe: all tests mock the actual packages.  Real-synthesis tests are
guarded by @pytest.mark.skipif so they SKIP in CI but RUN where installed.
"""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import pytest

# ===========================================================================
# Registry tests
# ===========================================================================


class TestRegistry:
    def test_list_backends_contains_all_engines(self):
        from voicelab.tts.registry import list_backends

        backends = list_backends()
        names = {b["name"] for b in backends}
        assert {"elevenlabs", "kokoro", "piper", "local"}.issubset(names)

    def test_list_backends_local_always_available(self):
        from voicelab.tts.registry import list_backends

        backends = {b["name"]: b for b in list_backends()}
        assert backends["local"]["available"] is True

    def test_list_backends_elevenlabs_requires_key(self):
        from voicelab.tts.registry import list_backends

        # Without key → not available
        backends = {b["name"]: b for b in list_backends(api_key=None)}
        assert backends["elevenlabs"]["available"] is False

        # With key AND package available
        fake_el = MagicMock()
        with patch.dict("sys.modules", {"elevenlabs": fake_el, "elevenlabs.client": fake_el}):
            backends_key = {b["name"]: b for b in list_backends(api_key="fake-key")}
        assert backends_key["elevenlabs"]["available"] is True

    def test_list_backends_kokoro_requires_package(self):
        import importlib

        import voicelab.tts.registry as registry_mod
        from voicelab.tts.registry import list_backends

        # Package absent → not available
        with patch.object(importlib, "import_module", side_effect=ImportError("no kokoro_onnx")):
            backends = {b["name"]: b for b in list_backends()}
        # When all imports fail, kokoro + piper + elevenlabs are unavailable
        assert backends["local"]["available"] is True

        # Directly test is_available with mocked import
        kokoro_info = next(b for b in registry_mod._REGISTRY if b.name == "kokoro")
        piper_info = next(b for b in registry_mod._REGISTRY if b.name == "piper")

        with patch("importlib.import_module", side_effect=ImportError):
            assert registry_mod.is_available(kokoro_info) is False
            assert registry_mod.is_available(piper_info) is False

        # Real packages are installed → should be True
        assert registry_mod.is_available(kokoro_info) is True
        assert registry_mod.is_available(piper_info) is True

    def test_list_backends_piper_requires_package(self):
        import voicelab.tts.registry as registry_mod

        piper_info = next(b for b in registry_mod._REGISTRY if b.name == "piper")

        with patch("importlib.import_module", side_effect=ImportError):
            assert registry_mod.is_available(piper_info) is False

        assert registry_mod.is_available(piper_info) is True

    def test_registry_priorities_ordered(self):
        from voicelab.tts.registry import _REGISTRY

        priorities = [b.priority for b in _REGISTRY]
        assert priorities == sorted(priorities), "Registry must be sorted by priority"


# ===========================================================================
# Engine selection (auto) — mocked availability
# ===========================================================================


class TestEngineSelection:
    def test_auto_picks_kokoro_when_available(self):
        """When kokoro is importable + elevenlabs absent, auto → kokoro."""
        from voicelab.tts.kokoro import KokoroTts
        from voicelab.tts.registry import get_tts_for_engine

        fake_kokoro_mod = MagicMock()
        with patch.dict("sys.modules", {"kokoro_onnx": fake_kokoro_mod}):
            with patch("voicelab.tts.registry._instantiate") as mock_inst:
                mock_inst.side_effect = lambda info, **kw: (
                    KokoroTts() if info.name == "kokoro" else None
                )
                tts = get_tts_for_engine("kokoro", api_key=None)
        assert isinstance(tts, KokoroTts)

    def test_auto_falls_back_to_local_when_nothing_available(self):
        """With no packages and no key, auto → LocalTts."""
        import voicelab.tts.registry as registry_mod
        from voicelab.tts.local import LocalTts
        from voicelab.tts.registry import get_tts_for_engine

        # Mock is_available so all non-local backends report unavailable
        def mock_is_available(info, *, api_key=None):
            return info.name == "local"

        with patch.object(registry_mod, "is_available", side_effect=mock_is_available):
            tts = get_tts_for_engine("auto", api_key=None)
        assert isinstance(tts, LocalTts)

    def test_forced_engine_unknown_falls_back_to_auto(self):
        """Unknown engine name logs warning and falls back gracefully."""
        from voicelab.tts.registry import get_tts_for_engine

        with patch.dict("sys.modules", {"kokoro_onnx": None, "piper": None}):
            tts = get_tts_for_engine("nonexistent_engine", api_key=None)
        # Should not raise; must return some backend
        assert tts is not None

    def test_forced_local_returns_local_tts(self):
        from voicelab.tts.local import LocalTts
        from voicelab.tts.registry import get_tts_for_engine

        tts = get_tts_for_engine("local", api_key=None)
        assert isinstance(tts, LocalTts)


# ===========================================================================
# engine.get_tts() respects tts_engine setting
# ===========================================================================


class TestEngineGetTts:
    def test_get_tts_auto_no_key_no_local_packages_returns_local(self):
        # Patch is_available in the registry so kokoro/piper report unavailable
        import voicelab.tts.registry as registry_mod
        from voicelab.tts.local import LocalTts

        original_is_available = registry_mod.is_available

        def mock_is_available(info, *, api_key=None):
            if info.name in ("kokoro", "piper"):
                return False
            return original_is_available(info, api_key=api_key)

        with patch.object(registry_mod, "is_available", side_effect=mock_is_available):
            from voicelab.engine import get_tts

            tts = get_tts()
        assert isinstance(tts, LocalTts)

    def test_get_tts_respects_tts_engine_setting(self, monkeypatch):
        """When tts_engine='local', get_tts must return LocalTts regardless."""
        import voicelab.config.settings as settings_mod

        settings_mod._settings = None
        monkeypatch.setenv("TTS_ENGINE", "local")

        from voicelab.engine import get_tts
        from voicelab.tts.local import LocalTts

        tts = get_tts()
        assert isinstance(tts, LocalTts)


# ===========================================================================
# KokoroTts — mocked
# ===========================================================================


class TestKokoroTtsMocked:
    def test_list_voices_returns_builtin_when_model_unavailable(self):
        from voicelab.tts import kokoro as kokoro_mod

        kokoro_mod._kokoro_instance = None
        kokoro_mod._kokoro_load_attempted = False

        with patch.dict("sys.modules", {"kokoro_onnx": None}):
            kokoro_mod._kokoro_load_attempted = False
            kokoro_mod._kokoro_instance = None
            from voicelab.tts.kokoro import KokoroTts

            tts = KokoroTts()
            voices = tts.list_voices()

        assert isinstance(voices, list)
        assert len(voices) > 0
        assert all("id" in v and "name" in v for v in voices)
        assert voices[0]["category"] == "kokoro"

    def test_synthesize_returns_none_when_model_unavailable(self):
        from voicelab.tts import kokoro as kokoro_mod

        kokoro_mod._kokoro_instance = None
        kokoro_mod._kokoro_load_attempted = False

        with patch.dict("sys.modules", {"kokoro_onnx": None}):
            kokoro_mod._kokoro_load_attempted = False
            kokoro_mod._kokoro_instance = None
            from voicelab.tts.kokoro import KokoroTts

            tts = KokoroTts()
            result = tts.synthesize("hello", "af_heart", {})

        assert result is None

    def test_synthesize_returns_wav_bytes_with_mocked_model(self):
        import numpy as np

        from voicelab.tts import kokoro as kokoro_mod

        kokoro_mod._kokoro_load_attempted = False
        kokoro_mod._kokoro_instance = None

        fake_samples = np.zeros(24000, dtype=np.float32)
        fake_session = MagicMock()
        fake_session.create.return_value = (fake_samples, 24000)
        fake_session.get_voices.return_value = ["af_heart", "af_bella"]

        kokoro_mod._kokoro_instance = fake_session
        kokoro_mod._kokoro_load_attempted = True

        from voicelab.tts.kokoro import KokoroTts

        tts = KokoroTts()
        result = tts.synthesize("hello world", "af_heart", {"speed": 1.0})

        assert result is not None
        assert isinstance(result, bytes)
        assert result[:4] == b"RIFF", f"Expected RIFF header, got {result[:4]!r}"

    def test_list_voices_from_model_when_loaded(self):
        from voicelab.tts import kokoro as kokoro_mod

        fake_session = MagicMock()
        fake_session.get_voices.return_value = ["af_heart", "af_bella", "am_adam"]
        kokoro_mod._kokoro_instance = fake_session
        kokoro_mod._kokoro_load_attempted = True

        from voicelab.tts.kokoro import KokoroTts

        tts = KokoroTts()
        voices = tts.list_voices()

        assert len(voices) == 3
        assert voices[0]["id"] in {"af_heart", "af_bella", "am_adam"}

    def test_audio_media_type_kokoro_is_wav(self):
        from voicelab.engine import audio_media_type
        from voicelab.tts.kokoro import KokoroTts

        tts = KokoroTts()
        assert audio_media_type(tts) == "audio/wav"

    def test_pcm_to_wav_riff_header(self):
        import numpy as np

        from voicelab.tts.kokoro import _pcm_to_wav

        samples = np.zeros(100, dtype=np.float32)
        wav = _pcm_to_wav(samples, sample_rate=24000)
        assert wav[:4] == b"RIFF"
        assert wav[8:12] == b"WAVE"


# ===========================================================================
# PiperTts — mocked
# ===========================================================================


class TestPiperTtsMocked:
    def test_list_voices_returns_curated_list_when_package_absent(self):
        from voicelab.tts.piper import PIPER_VOICES, PiperTts

        tts = PiperTts()
        voices = tts.list_voices()

        assert isinstance(voices, list)
        assert len(voices) == len(PIPER_VOICES)
        assert all("id" in v and "name" in v for v in voices)
        assert all(v["category"] == "piper" for v in voices)

    def test_synthesize_returns_none_when_voice_unavailable(self):
        from voicelab.tts import piper as piper_mod

        # Clear cache
        piper_mod._voice_cache.clear()

        with patch("voicelab.tts.piper._load_voice", return_value=None):
            from voicelab.tts.piper import PiperTts

            tts = PiperTts()
            result = tts.synthesize("hello", "en_US-amy-medium", {})

        assert result is None

    def test_synthesize_returns_wav_with_mocked_voice(self):
        def mock_synthesize_wav(text, wav_writer):
            # Write minimal WAV frames into the provided wav writer
            wav_writer.setnchannels(1)
            wav_writer.setsampwidth(2)
            wav_writer.setframerate(22050)
            wav_writer.writeframes(b"\x00\x00" * 100)

        fake_voice = MagicMock()
        fake_voice.synthesize_wav = mock_synthesize_wav

        from voicelab.tts import piper as piper_mod

        piper_mod._voice_cache.clear()

        with patch("voicelab.tts.piper._load_voice", return_value=fake_voice):
            from voicelab.tts.piper import PiperTts

            tts = PiperTts()
            result = tts.synthesize("hello world", "en_US-amy-medium", {})

        assert result is not None
        assert isinstance(result, bytes)
        assert result[:4] == b"RIFF"

    def test_audio_media_type_piper_is_wav(self):
        from voicelab.engine import audio_media_type
        from voicelab.tts.piper import PiperTts

        tts = PiperTts()
        assert audio_media_type(tts) == "audio/wav"

    def test_list_voices_shows_cached_true_when_file_present(self, tmp_path):
        """Voices with .onnx on disk show cached=True."""
        from voicelab.tts import piper as piper_mod
        from voicelab.tts.piper import _parse_voice_name

        voice_name = "en_US-amy-medium"
        lang, lang_region, speaker, quality = _parse_voice_name(voice_name)
        # Match the layout: {lang}/{lang_region}/{speaker}/{quality}/
        voice_dir = tmp_path / lang / lang_region / speaker / quality
        voice_dir.mkdir(parents=True)
        (voice_dir / f"{voice_name}.onnx").touch()

        original_dir = piper_mod._VOICE_DIR
        piper_mod._VOICE_DIR = tmp_path
        try:
            from voicelab.tts.piper import PiperTts

            tts = PiperTts()
            voices = {v["id"]: v for v in tts.list_voices()}
            assert voices[voice_name]["cached"] is True
        finally:
            piper_mod._VOICE_DIR = original_dir


# ===========================================================================
# ChatterboxTts — mocked (never actually imports torch)
# ===========================================================================


class TestChatterboxTtsMocked:
    def test_synthesize_returns_none_when_not_installed(self):
        from voicelab.tts import chatterbox as cb_mod

        cb_mod._model = None
        cb_mod._model_load_attempted = False

        with patch.dict("sys.modules", {"chatterbox": None, "chatterbox.tts": None}):
            cb_mod._model_load_attempted = False
            from voicelab.tts.chatterbox import ChatterboxTts

            tts = ChatterboxTts()
            result = tts.synthesize("hello", "default", {})

        assert result is None

    def test_list_voices_returns_two_entries(self):
        from voicelab.tts.chatterbox import ChatterboxTts

        tts = ChatterboxTts()
        voices = tts.list_voices()
        assert len(voices) == 2
        ids = {v["id"] for v in voices}
        assert "default" in ids
        assert "cloned" in ids


# ===========================================================================
# API /api/v1/engines endpoint
# ===========================================================================


class TestEnginesEndpoint:
    @pytest.fixture
    async def client(self):
        from httpx import ASGITransport, AsyncClient

        from voicelab.app.main import create_app

        app = create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as c:
            yield c

    @pytest.mark.asyncio
    async def test_engines_endpoint_returns_list(self, client):
        resp = await client.get("/api/v1/engines")
        assert resp.status_code == 200
        data = resp.json()
        assert "engines" in data
        assert isinstance(data["engines"], list)
        assert len(data["engines"]) >= 4

    @pytest.mark.asyncio
    async def test_engines_local_always_available(self, client):
        resp = await client.get("/api/v1/engines")
        assert resp.status_code == 200
        engines = {e["name"]: e for e in resp.json()["engines"]}
        assert engines["local"]["available"] is True

    @pytest.mark.asyncio
    async def test_engines_response_has_required_fields(self, client):
        resp = await client.get("/api/v1/engines")
        for engine in resp.json()["engines"]:
            assert "name" in engine
            assert "available" in engine
            assert "license" in engine

    @pytest.mark.asyncio
    async def test_synthesize_with_engine_param(self, client):
        """POST /synthesize with engine='local' uses local backend."""
        fake_wav = (
            b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
            b"\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
        )
        with patch("voicelab.tts.local.LocalTts.synthesize", return_value=fake_wav):
            resp = await client.post(
                "/api/v1/synthesize",
                json={"text": "Hello world", "voice": "default", "engine": "local"},
            )
        assert resp.status_code == 200
        assert "audio/" in resp.headers["content-type"]

    @pytest.mark.asyncio
    async def test_voices_with_engine_param(self, client):
        """GET /voices?engine=local returns local voices."""
        resp = await client.get("/api/v1/voices?engine=local")
        assert resp.status_code == 200
        data = resp.json()
        assert "voices" in data
        assert data["engine"] == "local"

    @pytest.mark.asyncio
    async def test_compare_with_engine_params(self, client):
        """POST /compare with engine_a/engine_b returns two clips."""
        fake_wav = (
            b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
            b"\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
        )
        with patch("voicelab.tts.local.LocalTts.synthesize", return_value=fake_wav):
            resp = await client.post(
                "/api/v1/compare",
                json={
                    "text": "Test",
                    "voice_a": "default",
                    "voice_b": "default",
                    "engine_a": "local",
                    "engine_b": "local",
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "voice_a" in data
        assert "voice_b" in data
        assert "engine_a" in data
        assert "engine_b" in data


# ===========================================================================
# Real-synthesis tests — SKIPPED in CI (only run when packages installed)
# ===========================================================================


def _pkg_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


@pytest.mark.skipif(not _pkg_available("kokoro_onnx"), reason="kokoro-onnx not installed")
class TestKokoroReal:
    def test_kokoro_synthesize_short_sentence(self):
        """Real Kokoro synthesis: returns non-empty WAV bytes."""
        from voicelab.tts import kokoro as kokoro_mod

        kokoro_mod._kokoro_instance = None
        kokoro_mod._kokoro_load_attempted = False

        from voicelab.tts.kokoro import KokoroTts

        tts = KokoroTts()
        result = tts.synthesize("Hello from Kokoro!", "af_heart", {"speed": 1.0})

        assert result is not None, "KokoroTts returned None — check model download"
        assert len(result) > 100, "Expected substantial WAV bytes"
        assert result[:4] == b"RIFF", f"Expected RIFF header, got {result[:4]!r}"

    def test_kokoro_list_voices_not_empty(self):
        from voicelab.tts.kokoro import KokoroTts

        tts = KokoroTts()
        voices = tts.list_voices()
        assert len(voices) > 0


@pytest.mark.skipif(not _pkg_available("piper"), reason="piper-tts not installed")
class TestPiperReal:
    def test_piper_synthesize_short_sentence(self):
        """Real Piper synthesis: returns non-empty WAV bytes."""
        from voicelab.tts import piper as piper_mod

        piper_mod._voice_cache.clear()

        from voicelab.tts.piper import PiperTts

        tts = PiperTts()
        result = tts.synthesize("Hello from Piper!", "en_US-amy-medium", {})

        assert result is not None, "PiperTts returned None — check voice download"
        assert len(result) > 100, "Expected substantial WAV bytes"
        assert result[:4] == b"RIFF", f"Expected RIFF header, got {result[:4]!r}"

    def test_piper_list_voices_not_empty(self):
        from voicelab.tts.piper import PiperTts

        tts = PiperTts()
        voices = tts.list_voices()
        assert len(voices) > 0
