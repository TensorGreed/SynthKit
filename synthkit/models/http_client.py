"""HTTP client that talks to OpenAI-compatible /chat/completions endpoints."""

from __future__ import annotations

from typing import List

import requests

from .client_base import ChatClient, ChatMessage
from ..config import ProviderConfig


class HTTPChatClient(ChatClient):
    """Generic wrapper for OSS or proxy deployments (vLLM, llamafile, etc.)."""

    def __init__(self, provider: ProviderConfig, model_name: str):
        self._cfg = provider
        self._model_name = model_name

    def chat(
        self,
        messages: List[ChatMessage],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Send the request to the configured base URL."""
        url = self._cfg.api_base.rstrip("/") + "/chat/completions"
        headers = {
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model_name,
            "messages": [message.__dict__ for message in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
