"""Shared structures used by curation judges."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class JudgedItem:
    """Normalized representation of an LLM inspection result."""

    score: float
    keep: bool
    label: str
    rationale: str
    original: Dict[str, Any]
