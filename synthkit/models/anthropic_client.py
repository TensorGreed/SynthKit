from __future__ import annotations
import os
from typing import List
import requests

from .client_base import ChatClient, ChatMessage
from ..config import ProviderConfig


class AnthropicChatClient(ChatClient):
    def __init__(self, provider: ProviderConfig, model_name: str):
        self._cfg = provider
        self._model_name = model_name
        api_key_env = provider.api_key_env or "ANTHROPIC_API_KEY"
        self._api_key = os.environ.get(api_key_env)
        if not self._api_key:
            raise RuntimeError(f"Missing API key env var: {api_key_env}")

    def chat(
        self,
        messages: List[ChatMessage],
        temperature: float,
        max_tokens: int,
    ) -> str:
        # Anthropic "messages" API style
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
            "messages": [
                {"role": m.role, "content": m.content} for m in messages if m.role != "system"
            ],
            "system": "\n".join(m.content for m in messages if m.role == "system"),
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return "".join(part["text"] for part in data["content"] if part["type"] == "text")
