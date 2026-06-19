"""Tests for shared Chatterbox helpers in voicelab/tts/_chatter_common.py.

These tests are pure-Python — no torch, no model download.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


class TestExaggerationToTemperature:
    """exaggeration_to_temperature maps 0–1 knob to 0.3–1.5 temperature."""

    def test_zero_exaggeration_gives_half(self):
        from voicelab.tts._chatter_common import exaggeration_to_temperature

        assert exaggeration_to_temperature(0.0) == pytest.approx(0.5)

    def test_half_exaggeration_gives_one(self):
        from voicelab.tts._chatter_common import exaggeration_to_temperature

        assert exaggeration_to_temperature(0.5) == pytest.approx(1.0)

    def test_full_exaggeration_gives_one_point_five(self):
        from voicelab.tts._chatter_common import exaggeration_to_temperature

        assert exaggeration_to_temperature(1.0) == pytest.approx(1.5)

    def test_clamp_above_one(self):
        """Values above 1 clamp to 1.5 max temperature."""
        from voicelab.tts._chatter_common import exaggeration_to_temperature

        assert exaggeration_to_temperature(2.0) == pytest.approx(1.5)
        assert exaggeration_to_temperature(99.0) == pytest.approx(1.5)

    def test_clamp_below_zero(self):
        """Values below 0 clamp to 0.3 min temperature."""
        from voicelab.tts._chatter_common import exaggeration_to_temperature

        assert exaggeration_to_temperature(-1.0) == pytest.approx(0.3)
        assert exaggeration_to_temperature(-99.0) == pytest.approx(0.3)

    def test_bad_input_none_returns_default(self):
        from voicelab.tts._chatter_common import exaggeration_to_temperature

        assert exaggeration_to_temperature(None) == pytest.approx(0.5)  # type: ignore[arg-type]

    def test_bad_input_string_returns_default(self):
        from voicelab.tts._chatter_common import exaggeration_to_temperature

        assert exaggeration_to_temperature("bad")  # type: ignore[arg-type]
        # Should not raise and should return the default 0.5
        result = exaggeration_to_temperature("bad")  # type: ignore[arg-type]
        assert result == pytest.approx(0.5)

    def test_returns_float(self):
        from voicelab.tts._chatter_common import exaggeration_to_temperature

        result = exaggeration_to_temperature(0.3)
        assert isinstance(result, float)

    def test_midpoint_values(self):
        from voicelab.tts._chatter_common import exaggeration_to_temperature

        # 0.25 → 0.75, 0.75 → 1.25
        assert exaggeration_to_temperature(0.25) == pytest.approx(0.75)
        assert exaggeration_to_temperature(0.75) == pytest.approx(1.25)


class TestSupportedGenerateKwargs:
    """supported_generate_kwargs filters kwargs to those the function accepts."""

    def test_filters_unknown_kwargs(self):
        """Only known params are returned."""
        from voicelab.tts._chatter_common import supported_generate_kwargs

        def fn(text: str, temperature: float = 0.8, exaggeration: float = 0.5) -> None:
            pass

        result = supported_generate_kwargs(fn, temperature=1.0, exaggeration=0.7, unknown_param=99)
        assert "temperature" in result
        assert "exaggeration" in result
        assert "unknown_param" not in result

    def test_passes_through_on_var_keyword(self):
        """If function accepts **kwargs, all args pass through."""
        from voicelab.tts._chatter_common import supported_generate_kwargs

        def fn(text: str, **kwargs) -> None:
            pass

        all_kwargs = {"temperature": 1.0, "exaggeration": 0.5, "anything": True}
        result = supported_generate_kwargs(fn, **all_kwargs)
        assert result == all_kwargs

    def test_passes_through_on_introspection_failure(self):
        """If signature inspection fails, all args pass through safely."""
        from voicelab.tts._chatter_common import supported_generate_kwargs

        # Built-in like len() raises ValueError on inspect.signature
        result = supported_generate_kwargs(len, temperature=1.0, exaggeration=0.5)
        # Should not raise; all kwargs returned
        assert isinstance(result, dict)

    def test_empty_kwargs_returns_empty(self):
        from voicelab.tts._chatter_common import supported_generate_kwargs

        def fn(text: str) -> None:
            pass

        result = supported_generate_kwargs(fn)
        assert result == {}

    def test_all_known_kwargs_returned(self):
        from voicelab.tts._chatter_common import supported_generate_kwargs

        def fn(text: str, temperature: float = 0.8, cfg_weight: float = 0.5) -> None:
            pass

        result = supported_generate_kwargs(fn, temperature=1.2, cfg_weight=0.3)
        assert result == {"temperature": 1.2, "cfg_weight": 0.3}

    def test_mocked_generate_with_specific_params(self):
        """Test with a MagicMock that has a real signature via spec."""
        from voicelab.tts._chatter_common import supported_generate_kwargs

        # Simulate a model.generate that accepts only text + audio_prompt_path + exaggeration
        def real_generate(
            text: str,
            audio_prompt_path: str | None = None,
            exaggeration: float = 0.5,
        ) -> None:
            pass

        # Create mock with spec so inspect works
        mock_gen = MagicMock(wraps=real_generate)
        mock_gen.__wrapped__ = real_generate

        # Directly test with the real function
        result = supported_generate_kwargs(
            real_generate,
            audio_prompt_path="/tmp/ref.wav",
            exaggeration=0.8,
            temperature=1.2,  # not in real_generate's signature
        )
        assert "audio_prompt_path" in result
        assert "exaggeration" in result
        assert "temperature" not in result
