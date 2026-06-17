"""Structured JSON logging configuration for voicelab.

Usage::

    from voicelab.logging_config import configure_logging
    configure_logging("DEBUG")
"""

from __future__ import annotations

import logging
import logging.config
import re
from typing import Any

# ---------------------------------------------------------------------------
# Secret redaction
# ---------------------------------------------------------------------------

_REDACT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"sk-[A-Za-z0-9_\-]{10,}", re.IGNORECASE),
    re.compile(r"Bearer\s+[A-Za-z0-9_\-]{10,}", re.IGNORECASE),
    re.compile(r"(?:key|token)=[A-Za-z0-9_\-]{20,}", re.IGNORECASE),
]

_REDACTED = "[REDACTED]"


def _redact(text: str) -> str:
    for pattern in _REDACT_PATTERNS:
        text = pattern.sub(_REDACTED, text)
    return text


class SecretRedactionFilter(logging.Filter):
    """Redacts API keys and bearer tokens from log records.

    Only redacts string arguments — numeric/other types are left untouched to
    avoid breaking format specifiers like ``%d`` in third-party log messages.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = _redact(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: _redact(v) if isinstance(v, str) else v for k, v in record.args.items()
                }
            else:
                record.args = tuple(_redact(a) if isinstance(a, str) else a for a in record.args)
        return True


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------

try:
    from pythonjsonlogger.json import JsonFormatter as _PJLFormatter

    class _JsonFormatter(_PJLFormatter):
        def add_fields(
            self,
            log_record: dict[str, Any],
            record: logging.LogRecord,
            message_dict: dict[str, Any],
        ) -> None:
            super().add_fields(log_record, record, message_dict)
            asctime_default = record.asctime if hasattr(record, "asctime") else ""
            log_record["time"] = log_record.pop("asctime", asctime_default)
            log_record["level"] = log_record.pop("levelname", record.levelname)
            log_record["logger"] = log_record.pop("name", record.name)
            log_record["message"] = log_record.pop("message", record.getMessage())
            log_record.setdefault("request_id", getattr(record, "request_id", None))

    _FORMATTER_CLASS = "voicelab.logging_config._JsonFormatter"
    _FORMATTER_KWARGS: dict[str, Any] = {"fmt": "%(asctime)s %(levelname)s %(name)s %(message)s"}

except ImportError:
    import json

    class _FallbackJsonFormatter(logging.Formatter):
        """Minimal JSON formatter used when python-json-logger is not installed."""

        def format(self, record: logging.LogRecord) -> str:
            record.message = record.getMessage()
            payload = {
                "time": self.formatTime(record, self.datefmt),
                "level": record.levelname,
                "logger": record.name,
                "message": record.message,
                "request_id": getattr(record, "request_id", None),
            }
            if record.exc_info:
                payload["exc_info"] = self.formatException(record.exc_info)
            return json.dumps(payload)

    _FORMATTER_CLASS = "voicelab.logging_config._FallbackJsonFormatter"
    _FORMATTER_KWARGS: dict[str, Any] = {}  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def configure_logging(level: str = "INFO") -> None:
    """Configure root and voicelab loggers with JSON output and secret redaction."""
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "filters": {
                "redact_secrets": {
                    "()": "voicelab.logging_config.SecretRedactionFilter",
                },
            },
            "formatters": {
                "json": {
                    "()": _FORMATTER_CLASS,
                    **_FORMATTER_KWARGS,
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                    "formatter": "json",
                    "filters": ["redact_secrets"],
                },
            },
            "root": {
                "level": level.upper(),
                "handlers": ["console"],
            },
            "loggers": {
                "voicelab": {
                    "level": level.upper(),
                    "handlers": ["console"],
                    "propagate": False,
                },
            },
        }
    )
