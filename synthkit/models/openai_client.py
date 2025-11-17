"""OpenAI native client that satisfies the shared chat protocol."""

from __future__ import annotations

import os
from typing import List, Optional

import requests
from requests import Session
from requests.exceptions import RequestException

from .client_base import ChatClient, ChatMessage, ChatClientError
from ..config import ProviderConfig
from .session import create_retry_session


class OpenAIChatClient(ChatClient):
    """Light wrapper around OpenAI's /chat/completions endpoint."""

    def __init__(
        self,
        provider: ProviderConfig,
        model_name: str,
        *,
        session: Optional[Session] = None,
    ):
        self._cfg = provider
        self._model_name = model_name
        api_key_env = provider.api_key_env or "OPENAI_API_KEY"
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
        """Perform a chat completion request using the configured credentials."""
        url = self._cfg.api_base.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model_name,
            "messages": [message.__dict__ for message in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            resp = self._session.post(url, json=payload, headers=headers, timeout=60)
            resp.raise_for_status()
        except RequestException as exc:  # pragma: no cover - network failure
            status = getattr(exc.response, "status_code", None)
            raise ChatClientError(
                provider="openai",
                model=self._model_name,
                message=str(exc),
                status_code=status,
            ) from exc
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def close(self) -> None:
        self._session.close()
