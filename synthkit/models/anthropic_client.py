"""Anthropic Claude client adapted to the shared ``ChatClient`` protocol."""

from __future__ import annotations

import os
from typing import List, Optional

import requests
from requests import Session
from requests.exceptions import RequestException

from .client_base import ChatClient, ChatMessage, ChatClientError
from ..config import ProviderConfig
from .session import create_retry_session


class AnthropicChatClient(ChatClient):
    """Wrap the Claude Messages API with the SynthKit client protocol."""

    def __init__(
        self,
        provider: ProviderConfig,
        model_name: str,
        *,
        session: Optional[Session] = None,
    ):
        self._cfg = provider
        self._model_name = model_name
        api_key_env = provider.api_key_env or "ANTHROPIC_API_KEY"
        self._api_key = os.environ.get(api_key_env)
        if not self._api_key:
            raise RuntimeError(f"Missing API key env var: {api_key_env}")
        self._session = session or create_retry_session()

    def chat(
        self,
        messages: List[ChatMessage],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call the Claude Messages endpoint and concatenate text parts."""
        url = self._cfg.api_base.rstrip("/") + "/messages"
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self._model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            # Anthropic separates system prompts from user/assistant history.
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
                if message.role != "system"
            ],
            "system": "\n".join(message.content for message in messages if message.role == "system"),
        }
        try:
            resp = self._session.post(url, json=payload, headers=headers, timeout=60)
            resp.raise_for_status()
        except RequestException as exc:  # pragma: no cover - network failure
            status = getattr(exc.response, "status_code", None)
            raise ChatClientError(
                provider="anthropic",
                model=self._model_name,
                message=str(exc),
                status_code=status,
            ) from exc
        data = resp.json()
        return "".join(part["text"] for part in data["content"] if part["type"] == "text")

    def close(self) -> None:
        """Release the underlying HTTP session."""
        self._session.close()
