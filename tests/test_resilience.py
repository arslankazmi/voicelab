"""Tests for voicelab.resilience and ElevenLabsTts resilient wrappers.

All tests are keyless — no real ElevenLabs API key required.
"""

from __future__ import annotations

import sys
import time
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeHttpError(Exception):
    """Simulates an SDK HTTP error with a status code attribute."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


# ---------------------------------------------------------------------------
# resilient_call — unit tests
# ---------------------------------------------------------------------------


class TestResilientCall:
    """Tests for the core resilience wrapper."""

    def test_success_on_first_attempt(self):
        """fn that succeeds immediately returns the value."""
        from voicelab.resilience import resilient_call

        result = resilient_call(lambda: 42, timeout=5.0, retries=2)
        assert result == 42

    def test_retries_transient_then_succeeds(self):
        """fn that raises TimeoutError twice then succeeds — call succeeds."""
        from voicelab.resilience import resilient_call

        call_count = 0

        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TimeoutError("simulated timeout")
            return "done"

        result = resilient_call(flaky, timeout=5.0, retries=2, backoff_base=0.01)
        assert result == "done"
        assert call_count == 3

    def test_gives_up_after_n_retries(self):
        """fn that always raises TimeoutError → TransientError after exhausting."""
        from voicelab.resilience import TransientError, resilient_call

        call_count = 0

        def always_timeout():
            nonlocal call_count
            call_count += 1
            raise TimeoutError("always")

        with pytest.raises(TransientError):
            resilient_call(always_timeout, timeout=5.0, retries=2, backoff_base=0.01)

        # 1 initial + 2 retries = 3 total
        assert call_count == 3

    def test_does_not_retry_non_transient_401(self):
        """fn raising 401 → NonTransientError immediately, no retry."""
        from voicelab.resilience import NonTransientError, resilient_call

        call_count = 0

        def auth_fail():
            nonlocal call_count
            call_count += 1
            raise _FakeHttpError(401)

        with pytest.raises((NonTransientError, _FakeHttpError)):
            resilient_call(auth_fail, timeout=5.0, retries=2, backoff_base=0.01)

        # Should not retry; attempt count = 1
        assert call_count == 1

    def test_does_not_retry_non_transient_403(self):
        """fn raising 403 → does not retry."""
        from voicelab.resilience import NonTransientError, resilient_call

        call_count = 0

        def forbidden():
            nonlocal call_count
            call_count += 1
            raise _FakeHttpError(403)

        with pytest.raises((NonTransientError, _FakeHttpError)):
            resilient_call(forbidden, timeout=5.0, retries=2, backoff_base=0.01)

        assert call_count == 1

    def test_retries_500_transient(self):
        """fn raising 500 is retried (server error = transient)."""
        from voicelab.resilience import resilient_call

        call_count = 0

        def server_error():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise _FakeHttpError(500)
            return "recovered"

        result = resilient_call(server_error, timeout=5.0, retries=2, backoff_base=0.01)
        assert result == "recovered"
        assert call_count == 2

    def test_respects_timeout(self):
        """fn that sleeps longer than timeout → TimeoutError/TransientError."""
        from voicelab.resilience import TransientError, resilient_call

        def slow():
            time.sleep(10)  # much longer than timeout
            return "never"

        # Very short timeout; retries=0 so only one attempt
        with pytest.raises((TransientError, TimeoutError)):
            resilient_call(slow, timeout=0.05, retries=0, backoff_base=0.01)


# ---------------------------------------------------------------------------
# ElevenLabsTts with mocked SDK — resilience integration
# ---------------------------------------------------------------------------


def _make_fake_sdk(synthesize_side_effect=None, voices_side_effect=None):
    """Build a fake elevenlabs SDK module hierarchy."""
    fake_voice_settings_cls = MagicMock()
    fake_sdk = MagicMock()
    fake_sdk.VoiceSettings = fake_voice_settings_cls

    fake_client_module = MagicMock()
    fake_el_instance = MagicMock()

    if synthesize_side_effect is not None:
        fake_el_instance.text_to_speech.convert.side_effect = synthesize_side_effect

    if voices_side_effect is not None:
        fake_el_instance.voices.get_all.side_effect = voices_side_effect

    fake_client_cls = MagicMock(return_value=fake_el_instance)
    fake_client_module.ElevenLabs = fake_client_cls

    fake_sdk.client = fake_client_module

    return fake_sdk, fake_el_instance


class TestElevenLabsTtsResilient:
    """Tests that mocked EL SDK calls are wrapped with resilience."""

    def _make_tts(self, fake_sdk, fake_el_instance):
        """Construct ElevenLabsTts with the fake SDK patched in."""
        with patch.dict(
            sys.modules,
            {
                "elevenlabs": fake_sdk,
                "elevenlabs.client": fake_sdk.client,
            },
        ):
            from voicelab.tts.elevenlabs import ElevenLabsTts

            tts = ElevenLabsTts(
                api_key="fake-key",
                timeout=5.0,
                retries=2,
                backoff_base=0.01,
            )
            # Attach the live instance so tests can inspect it
            tts._client = fake_el_instance
        return tts

    def test_synthesize_retries_twice_then_succeeds(self):
        """Synthesize with SDK raising TransientError twice then returning audio."""
        fake_sdk, fake_el = _make_fake_sdk()
        tts = self._make_tts(fake_sdk, fake_el)

        call_count = 0
        fake_chunk = b"audio_chunk"

        def convert_side_effect(**_kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TimeoutError("simulated timeout")
            return iter([fake_chunk])

        fake_el.text_to_speech.convert.side_effect = convert_side_effect

        with patch.dict(sys.modules, {"elevenlabs": fake_sdk, "elevenlabs.client": fake_sdk.client}):
            result = tts.synthesize("hello", "voice_id_1", {})

        assert result == fake_chunk
        assert call_count == 3

    def test_synthesize_always_timeout_degrades_to_none(self):
        """Synthesize that always times out returns None without raising."""
        fake_sdk, fake_el = _make_fake_sdk()
        tts = self._make_tts(fake_sdk, fake_el)

        fake_el.text_to_speech.convert.side_effect = TimeoutError("always")

        with patch.dict(sys.modules, {"elevenlabs": fake_sdk, "elevenlabs.client": fake_sdk.client}):
            result = tts.synthesize("hello", "voice_id_1", {})

        assert result is None  # graceful degradation, never raises

    def test_synthesize_does_not_hang_event_loop(self):
        """Verify synthesize completes quickly even when SDK times out."""
        fake_sdk, fake_el = _make_fake_sdk()
        tts = self._make_tts(fake_sdk, fake_el)

        # retries=0, very short timeout — should complete fast
        tts._retries = 0
        tts._timeout = 0.05
        fake_el.text_to_speech.convert.side_effect = lambda **_: time.sleep(10)

        start = time.monotonic()
        with patch.dict(sys.modules, {"elevenlabs": fake_sdk, "elevenlabs.client": fake_sdk.client}):
            result = tts.synthesize("hello", "voice_id", {})
        elapsed = time.monotonic() - start

        assert result is None
        assert elapsed < 5.0, f"synthesize took {elapsed:.2f}s — should have timed out fast"

    def test_list_voices_success(self):
        """list_voices returns parsed list when SDK succeeds."""
        fake_sdk, fake_el = _make_fake_sdk()
        tts = self._make_tts(fake_sdk, fake_el)

        fake_voice = MagicMock()
        fake_voice.voice_id = "v1"
        fake_voice.name = "Rachel"
        fake_voice.category = "premade"
        fake_resp = MagicMock()
        fake_resp.voices = [fake_voice]
        fake_el.voices.get_all.return_value = fake_resp

        result = tts.list_voices()

        assert isinstance(result, list)
        assert result[0]["id"] == "v1"
        assert result[0]["name"] == "Rachel"

    def test_list_voices_always_fails_returns_empty_list(self):
        """list_voices degrades to [] when SDK always raises."""
        fake_sdk, fake_el = _make_fake_sdk()
        tts = self._make_tts(fake_sdk, fake_el)

        fake_el.voices.get_all.side_effect = TimeoutError("always")

        result = tts.list_voices()
        assert result == []  # graceful degradation


# ---------------------------------------------------------------------------
# Metrics integration — backend/outcome labels
# ---------------------------------------------------------------------------


class TestTtsMetrics:
    """Verify tts_calls_total and tts_latency_seconds are incremented."""

    def _fresh_metrics(self):
        """Import metrics fresh (may be no-op if prometheus not installed)."""
        from voicelab.metrics import tts_calls_total, tts_latency_seconds

        return tts_calls_total, tts_latency_seconds

    def test_metrics_incremented_on_success(self):
        """Successful synthesize → tts_calls_total ok label incremented."""
        tts_calls_total, tts_latency_seconds = self._fresh_metrics()

        fake_sdk, fake_el = _make_fake_sdk()

        fake_chunk = b"audio"
        fake_el.text_to_speech.convert.return_value = iter([fake_chunk])

        # Patch labels() call to capture invocations
        incremented = []
        observed = []

        original_labels_calls = tts_calls_total.labels
        original_latency_labels = tts_latency_seconds.labels

        class CapturingCounter:
            def __init__(self, backend, outcome):
                self.backend = backend
                self.outcome = outcome

            def inc(self, amount=1):
                incremented.append((self.backend, self.outcome))

        class CapturingHistogram:
            def __init__(self, backend, outcome):
                self.backend = backend
                self.outcome = outcome

            def observe(self, val):
                observed.append((self.backend, self.outcome, val))

        with patch.object(
            tts_calls_total, "labels", side_effect=lambda **kw: CapturingCounter(**kw)
        ), patch.object(
            tts_latency_seconds, "labels", side_effect=lambda **kw: CapturingHistogram(**kw)
        ):
            with patch.dict(
                sys.modules,
                {"elevenlabs": fake_sdk, "elevenlabs.client": fake_sdk.client},
            ):
                from voicelab.tts.elevenlabs import ElevenLabsTts

                tts = ElevenLabsTts(api_key="fake-key", timeout=5.0, retries=0)
                tts._client = fake_el
                tts.synthesize("hello", "v1", {})

        assert any(b == "elevenlabs" and o == "ok" for b, o in incremented), (
            f"Expected (elevenlabs, ok) in tts_calls_total increments, got {incremented}"
        )
        assert any(b == "elevenlabs" and o == "ok" for b, o, _ in observed), (
            f"Expected (elevenlabs, ok) in tts_latency_seconds observations, got {observed}"
        )

    def test_metrics_incremented_on_error(self):
        """Failed synthesize → tts_calls_total error label incremented."""
        tts_calls_total, tts_latency_seconds = self._fresh_metrics()

        fake_sdk, fake_el = _make_fake_sdk()
        fake_el.text_to_speech.convert.side_effect = TimeoutError("always")

        incremented = []

        class CapturingCounter:
            def __init__(self, backend, outcome):
                self.backend = backend
                self.outcome = outcome

            def inc(self, amount=1):
                incremented.append((self.backend, self.outcome))

        class CapturingHistogram:
            def observe(self, val):
                pass

        with patch.object(
            tts_calls_total, "labels", side_effect=lambda **kw: CapturingCounter(**kw)
        ), patch.object(tts_latency_seconds, "labels", return_value=CapturingHistogram()):
            with patch.dict(
                sys.modules,
                {"elevenlabs": fake_sdk, "elevenlabs.client": fake_sdk.client},
            ):
                from voicelab.tts.elevenlabs import ElevenLabsTts

                tts = ElevenLabsTts(api_key="fake-key", timeout=5.0, retries=0)
                tts._client = fake_el
                result = tts.synthesize("hello", "v1", {})

        assert result is None
        assert any(b == "elevenlabs" and o == "error" for b, o in incremented), (
            f"Expected (elevenlabs, error) in increments, got {incremented}"
        )

    def test_metrics_labels_accept_backend_and_outcome(self):
        """tts_calls_total.labels(backend=..., outcome=...) does not raise."""
        tts_calls_total, tts_latency_seconds = self._fresh_metrics()
        # Should not raise regardless of prometheus install
        counter = tts_calls_total.labels(backend="elevenlabs", outcome="ok")
        assert counter is not None
        histogram = tts_latency_seconds.labels(backend="elevenlabs", outcome="error")
        assert histogram is not None
