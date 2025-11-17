"""Shared protocol definitions for chat-completion clients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol


@dataclass
class ChatMessage:
    """Simple container mirroring OpenAI-style chat messages."""

    role: str  # "system" | "user" | "assistant"
    content: str


class ChatClientError(RuntimeError):
    """Raised when an upstream provider request fails."""

    def __init__(
        self,
        provider: str,
        model: str,
        message: str,
        *,
        status_code: int | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.status_code = status_code
        self.message = message
        context = f"[{provider}:{model}] {message}"
        if status_code is not None:
            context = f"{context} (status={status_code})"
        super().__init__(context)


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

    def close(self) -> None:  # pragma: no cover - optional hook
        """Close any open resources (sockets, sessions)."""
        ...
