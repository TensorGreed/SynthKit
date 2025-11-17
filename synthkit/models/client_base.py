from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Protocol


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


class ChatClient(Protocol):
    """Minimal common chat interface."""

    def chat(
        self,
        messages: List[ChatMessage],
        temperature: float,
        max_tokens: int,
    ) -> str:
        ...
