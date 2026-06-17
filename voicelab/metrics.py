"""Prometheus custom metrics for voicelab.

Imported by API handlers and TTS backends to record per-call latency and
outcome. Gracefully no-ops if prometheus_client is not installed.

Usage::

    from voicelab.metrics import tts_calls_total, tts_latency_seconds
    tts_calls_total.labels(backend="elevenlabs", outcome="ok").inc()
    with tts_latency_seconds.labels(backend="elevenlabs", outcome="ok").time():
        ...

Labels
------
backend : str
    "elevenlabs" | "local" | …
outcome : str
    "ok" | "error"
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Histogram

    tts_calls_total: Counter = Counter(
        "tts_calls_total",
        "Total number of TTS synthesis calls.",
        labelnames=["backend", "outcome"],
    )

    tts_latency_seconds: Histogram = Histogram(
        "tts_latency_seconds",
        "TTS synthesis latency in seconds.",
        labelnames=["backend", "outcome"],
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )

except ImportError:
    logger.warning("prometheus_client not installed — TTS metrics are no-ops.")

    class _NoOpMetric:
        """Drop-in no-op replacement for Counter / Histogram."""

        def labels(self, **_kwargs: object) -> _NoOpMetric:
            return self

        def inc(self, _amount: float = 1) -> None:
            pass

        def observe(self, _amount: float) -> None:
            pass

        def time(self) -> _NoOpCtx:
            return _NoOpCtx()

    class _NoOpCtx:
        def __enter__(self) -> _NoOpCtx:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

    tts_calls_total = _NoOpMetric()  # type: ignore[assignment]
    tts_latency_seconds = _NoOpMetric()  # type: ignore[assignment]
