"""Ollama client implementation for the shared chat interface."""

from __future__ import annotations

from typing import List, Optional

from requests import Session
from requests.exceptions import RequestException

from .client_base import ChatClient, ChatMessage, ChatClientError
from ..config import ProviderConfig
from .session import create_retry_session


class OllamaChatClient(ChatClient):
    """Call a local/remote Ollama server via its /api/chat endpoint."""

    def __init__(
        self,
        provider: ProviderConfig,
        model_name: str,
        *,
        session: Optional[Session] = None,
    ):
        self._cfg = provider
        self._model_name = model_name
        self._session = session or create_retry_session()

    def chat(
        self,
        messages: List[ChatMessage],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Send chat messages to an Ollama server and return final text."""
        url = self._cfg.api_base.rstrip("/") + "/api/chat"
        payload = {
            "model": self._model_name,
            "messages": [message.__dict__ for message in messages],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        try:
            resp = self._session.post(url, json=payload, timeout=600)
            resp.raise_for_status()
        except RequestException as exc:  # pragma: no cover - network failure
            status = getattr(exc.response, "status_code", None)
            raise ChatClientError(
                provider="ollama",
                model=self._model_name,
                message=str(exc),
                status_code=status,
            ) from exc
        data = resp.json()
        message = data.get("message", {})
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
        content = data.get("response")
        if isinstance(content, str):
            return content
        raise ChatClientError(
            provider="ollama",
            model=self._model_name,
            message="Unexpected Ollama response payload",
        )

    def close(self) -> None:
        self._session.close()
