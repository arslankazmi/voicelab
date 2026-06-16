"""Tests for persona loading."""

from __future__ import annotations

import pytest

from voicelab.personas import load_personas, Persona, PersonaSettings


def test_personas_load_returns_list():
    personas = load_personas()
    assert isinstance(personas, list)
    assert len(personas) > 0


def test_personas_have_required_fields():
    for p in load_personas():
        assert isinstance(p, Persona)
        assert p.name, "persona.name must be non-empty"
        assert p.description, "persona.description must be non-empty"
        assert p.sample_line, "persona.sample_line must be non-empty"
        assert isinstance(p.default_settings, PersonaSettings)


def test_persona_settings_valid_ranges():
    for p in load_personas():
        s = p.default_settings
        assert 0.0 <= s.stability <= 1.0, f"{p.name}: stability out of range"
        assert 0.0 <= s.similarity_boost <= 1.0, f"{p.name}: similarity_boost out of range"
        assert 0.0 <= s.style <= 1.0, f"{p.name}: style out of range"
        assert 0.5 <= s.speed <= 2.0, f"{p.name}: speed out of range"


def test_personas_have_five_presets():
    personas = load_personas()
    assert len(personas) == 5


def test_persona_names_are_unique():
    names = [p.name for p in load_personas()]
    assert len(names) == len(set(names)), "Persona names must be unique"


def test_personas_cache():
    """load_personas() returns the same list object on repeated calls."""
    a = load_personas()
    b = load_personas()
    assert a is b
