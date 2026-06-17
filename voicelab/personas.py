"""Persona loading — reads personas.yaml and exposes Persona dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

_PERSONAS_YAML = Path(__file__).parent / "personas.yaml"


@dataclass
class PersonaSettings:
    stability: float = 0.75
    similarity_boost: float = 0.75
    style: float = 0.0
    speed: float = 1.0


@dataclass
class Persona:
    name: str
    description: str
    suggested_voice_traits: str
    default_settings: PersonaSettings
    sample_line: str

    @classmethod
    def from_dict(cls, data: dict) -> Persona:
        raw_settings = data.get("default_settings", {})
        settings = PersonaSettings(
            stability=float(raw_settings.get("stability", 0.75)),
            similarity_boost=float(raw_settings.get("similarity_boost", 0.75)),
            style=float(raw_settings.get("style", 0.0)),
            speed=float(raw_settings.get("speed", 1.0)),
        )
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            suggested_voice_traits=data.get("suggested_voice_traits", ""),
            default_settings=settings,
            sample_line=data.get("sample_line", ""),
        )


_personas: list[Persona] | None = None


def load_personas() -> list[Persona]:
    """Load persona presets from personas.yaml. Cached after first call."""
    global _personas
    if _personas is None:
        raw = yaml.safe_load(_PERSONAS_YAML.read_text()) or {}
        _personas = [Persona.from_dict(p) for p in raw.get("personas", [])]
    return _personas


def get_persona(name: str) -> Persona | None:
    """Lookup persona by name; returns None if not found."""
    return next((p for p in load_personas() if p.name == name), None)
