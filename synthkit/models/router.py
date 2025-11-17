"""Factory for memoizing chat clients keyed by logical stage references."""

from __future__ import annotations

from typing import Dict

from .client_base import ChatClient
from .openai_client import OpenAIChatClient
from .anthropic_client import AnthropicChatClient
from .http_client import HTTPChatClient
from ..config import ForgeConfig, ModelRef, ProviderConfig


def _build_client(provider_cfg: ProviderConfig, model_name: str) -> ChatClient:
    """Instantiate the correct provider-specific client."""
    if provider_cfg.type == "openai":
        return OpenAIChatClient(provider_cfg, model_name)
    if provider_cfg.type == "anthropic":
        return AnthropicChatClient(provider_cfg, model_name)
    if provider_cfg.type == "http":
        return HTTPChatClient(provider_cfg, model_name)
    raise ValueError(f"Unknown provider type: {provider_cfg.type}")


class ModelRouter:
    """Cache chat clients so each stage reuses HTTP sessions where possible."""

    def __init__(self, cfg: ForgeConfig):
        self._cfg = cfg
        self._cache: Dict[str, ChatClient] = {}

    def for_stage(self, ref: ModelRef) -> ChatClient:
        """Return (and memoize) the client for the requested model reference."""
        key = f"{ref.provider}:{ref.name}"
        if key in self._cache:
            return self._cache[key]

        provider_cfg = self._cfg.providers[ref.provider]
        client = _build_client(provider_cfg, ref.name)
        self._cache[key] = client
        return client

    def close_all(self) -> None:
        """Close all cached clients and clear the memoized map."""
        for client in self._cache.values():
            close = getattr(client, "close", None)
            if callable(close):
                close()
        self._cache.clear()
