from __future__ import annotations
import os
from typing import List
import requests

from .client_base import ChatClient, ChatMessage
from ..config import ProviderConfig


class OpenAIChatClient(ChatClient):
    def __init__(self, provider: ProviderConfig, model_name: str):
        self._cfg = provider
        self._model_name = model_name
        api_key_env = provider.api_key_env or "OPENAI_API_KEY"
        self._api_key = os.environ.get(api_key_env)
        if not self._api_key:
            raise RuntimeError(f"Missing API key env var: {api_key_env}")

    def chat(
        self,
        messages: List[ChatMessage],
        temperature: float,
        max_tokens: int,
    ) -> str:
        url = self._cfg.api_base.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model_name,
            "messages": [m.__dict__ for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
