"""Lightweight plugin registries for generators and formatters."""

from __future__ import annotations

from typing import Callable, Mapping, MutableMapping, Sequence

from .config import ForgeConfig
from .generation.base import BaseGenerator
from .models.client_base import ChatClient

GeneratorFactory = Callable[[ChatClient, ForgeConfig], BaseGenerator]
Formatter = Callable[[Mapping[str, object]], Mapping[str, object]]

_generator_registry: MutableMapping[str, GeneratorFactory] = {}
_formatter_registry: MutableMapping[str, Formatter] = {}


def register_generator(name: str, factory: GeneratorFactory, *, override: bool = False) -> None:
    """Register a generator factory under ``name``."""
    key = name.lower()
    if not override and key in _generator_registry:
        raise ValueError(f"Generator '{name}' already registered")
    _generator_registry[key] = factory


def get_generator_factory(name: str) -> GeneratorFactory:
    """Retrieve a generator factory by name."""
    try:
        return _generator_registry[name.lower()]
    except KeyError as exc:  # pragma: no cover - simple accessor
        raise KeyError(f"Unknown generator '{name}'. Available: {available_generator_types()}") from exc


def available_generator_types() -> Sequence[str]:
    """Return registered generator keys sorted alphabetically."""
    return tuple(sorted(_generator_registry.keys()))


def register_formatter(name: str, formatter: Formatter, *, override: bool = False) -> None:
    """Register a sample formatter under ``name``."""
    key = name.lower()
    if not override and key in _formatter_registry:
        raise ValueError(f"Formatter '{name}' already registered")
    _formatter_registry[key] = formatter


def get_formatter(name: str) -> Formatter:
    """Retrieve a formatter by name."""
    try:
        return _formatter_registry[name.lower()]
    except KeyError as exc:  # pragma: no cover - simple accessor
        raise KeyError(f"Unknown formatter '{name}'. Available: {available_formatter_names()}") from exc


def available_formatter_names() -> Sequence[str]:
    """Return registered formatter keys sorted alphabetically."""
    return tuple(sorted(_formatter_registry.keys()))
