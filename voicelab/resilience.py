"""Resilient wrapper for blocking SDK calls.

``resilient_call`` runs a blocking callable in a thread-pool with a hard
wall-clock timeout, retries transient failures (timeouts, 5xx, connection
errors) with exponential back-off via *tenacity*, and raises a clean typed
error after exhausting all attempts.

Non-transient failures (auth errors, 4xx) are NOT retried — they propagate
immediately as :class:`NonTransientError`.

Usage::

    from voicelab.resilience import resilient_call, TransientError, NonTransientError

    result = resilient_call(
        lambda: sdk.text_to_speech.convert(...),
        timeout=30.0,
        retries=2,
        backoff_base=0.5,
    )
"""

from __future__ import annotations

import concurrent.futures
import logging
from collections.abc import Callable

import tenacity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Typed errors
# ---------------------------------------------------------------------------


class TransientError(RuntimeError):
    """Raised after all retry attempts are exhausted on a transient failure."""


class NonTransientError(RuntimeError):
    """Raised immediately on a non-transient (auth / 4xx) failure."""


# ---------------------------------------------------------------------------
# Helpers — classify exceptions
# ---------------------------------------------------------------------------

_TRANSIENT_STATUS_CODES = frozenset(range(500, 600)) | {408, 429}


def _is_transient(exc: BaseException) -> bool:
    """Return True if *exc* should trigger a retry."""
    # TimeoutError covers both asyncio.TimeoutError and concurrent.futures variants
    if isinstance(exc, (TimeoutError, concurrent.futures.TimeoutError)):
        return True

    # Connection-level errors
    if isinstance(
        exc,
        (
            ConnectionError,
            ConnectionRefusedError,
            ConnectionResetError,
            BrokenPipeError,
            OSError,
        ),
    ):
        return True

    # HTTP-level status code — look for common SDK patterns
    status: int | None = None
    for attr in ("status_code", "status", "code", "http_status"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            status = val
            break
    # Also check __cause__ / __context__ for wrapped HTTP errors
    if status is None:
        for linked in (exc.__cause__, exc.__context__):
            if linked is not None:
                for attr in ("status_code", "status", "code", "http_status"):
                    val = getattr(linked, attr, None)
                    if isinstance(val, int):
                        status = val
                        break
            if status is not None:
                break

    if status is not None:
        return status in _TRANSIENT_STATUS_CODES

    return False


def _is_non_transient(exc: BaseException) -> bool:
    """Return True for auth / 4xx errors that must NOT be retried."""
    status: int | None = None
    for attr in ("status_code", "status", "code", "http_status"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            status = val
            break
    if status is None:
        for linked in (exc.__cause__, exc.__context__):
            if linked is not None:
                for attr in ("status_code", "status", "code", "http_status"):
                    val = getattr(linked, attr, None)
                    if isinstance(val, int):
                        status = val
                        break
            if status is not None:
                break

    if status is not None:
        return 400 <= status < 500 and status not in (408, 429)

    return False


# ---------------------------------------------------------------------------
# Core wrapper
# ---------------------------------------------------------------------------


def resilient_call[T](
    fn: Callable[[], T],
    *,
    timeout: float,
    retries: int = 2,
    backoff_base: float = 0.5,
) -> T:
    """Run *fn* in a thread-pool with timeout and retry logic.

    Args:
        fn: A zero-argument callable wrapping the blocking SDK call.
        timeout: Wall-clock seconds before a single attempt is cancelled.
        retries: Maximum number of *retry* attempts after the first failure
            (total attempts = retries + 1).
        backoff_base: Base seconds for exponential back-off between retries.

    Returns:
        Whatever *fn* returns.

    Raises:
        NonTransientError: Immediately on auth / 4xx without retrying.
        TransientError: After all retry attempts are exhausted.
    """
    _executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    @tenacity.retry(
        retry=tenacity.retry_if_exception(_is_transient),
        stop=tenacity.stop_after_attempt(retries + 1),
        wait=tenacity.wait_exponential(multiplier=backoff_base, min=backoff_base, max=30),
        reraise=False,
        before_sleep=lambda rs: logger.warning(
            "resilient_call: transient error on attempt %d/%d — retrying",
            rs.attempt_number,
            retries + 1,
        ),
    )
    def _attempt() -> T:
        future = _executor.submit(fn)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise TimeoutError(f"SDK call timed out after {timeout}s") from exc

    try:
        return _attempt()
    except tenacity.RetryError as exc:
        # Exhausted retries — unwrap last exception
        last = exc.last_attempt.exception()
        raise TransientError(f"SDK call failed after {retries + 1} attempts: {last}") from last
    except (NonTransientError, TransientError):
        raise
    except Exception as exc:
        # Non-transient at call time — raise immediately
        if _is_non_transient(exc):
            raise NonTransientError(f"Non-transient SDK error: {exc}") from exc
        # Otherwise it's unexpected; surface as transient after tenacity gives up
        raise TransientError(f"SDK call failed: {exc}") from exc
    finally:
        _executor.shutdown(wait=False)
