"""Shared protocol definitions for chat-completion clients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol


@dataclass
class ChatMessage:
    """Simple container mirroring OpenAI-style chat messages."""

    role: str  # "system" | "user" | "assistant"
    content: str


class ChatClient(Protocol):
    """Minimal interface implemented by each provider wrapper."""

    def chat(
        self,
        messages: List[ChatMessage],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Send chat messages and return the assistant text response."""
        ...
