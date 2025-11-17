from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class JudgedItem:
    score: float
    keep: bool
    label: str
    rationale: str
    original: Dict[str, Any]
