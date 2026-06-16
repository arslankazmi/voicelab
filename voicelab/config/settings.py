"""Application settings — pydantic-settings with config.yaml deep-merge.

Usage::

    from voicelab.config.settings import get_settings
    s = get_settings()
    print(s.app_port)

Environment variables override config.yaml values; config.yaml overrides _DEFAULTS.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

_DEFAULTS: dict[str, Any] = {
    "app_port": 8001,
}

_CONFIG_YAML = Path(__file__).parent.parent.parent / "config.yaml"


def _load_yaml_config() -> dict[str, Any]:
    """Deep-merge config.yaml over _DEFAULTS. Returns a flat dict."""
    merged: dict[str, Any] = dict(_DEFAULTS)
    if _CONFIG_YAML.exists():
        raw = yaml.safe_load(_CONFIG_YAML.read_text()) or {}
        # Flatten nested keys: app.port -> app_port
        for section, values in raw.items():
            if isinstance(values, dict):
                for key, val in values.items():
                    merged[f"{section}_{key}"] = val
            else:
                merged[section] = values
    return merged


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App
    app_port: int = _DEFAULTS["app_port"]

    # API keys — all Optional so the app boots without secrets
    elevenlabs_api_key: Optional[str] = None


_settings: Settings | None = None


def get_settings() -> Settings:
    """Lazy singleton — reads .env and config.yaml on first call."""
    global _settings
    if _settings is None:
        yaml_cfg = _load_yaml_config()
        _settings = Settings(**{k: v for k, v in yaml_cfg.items() if k in Settings.model_fields})
    return _settings
