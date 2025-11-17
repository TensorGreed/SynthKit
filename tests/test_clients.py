import os

import pytest
import requests

from synthkit.config import ProviderConfig
from synthkit.models.client_base import ChatMessage, ChatClientError
from synthkit.models.openai_client import OpenAIChatClient


class DummySession:
    def __init__(self, exception: Exception):
        self._exception = exception
        self.closed = False

    def post(self, *args, **kwargs):  # pragma: no cover - invoked in tests
        raise self._exception

    def close(self):  # pragma: no cover - invoked in tests
        self.closed = True


def test_openai_client_raises_chat_client_error(monkeypatch):
    provider = ProviderConfig(type="openai", api_base="https://example.com")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    session = DummySession(requests.exceptions.RequestException("boom"))
    client = OpenAIChatClient(provider, "gpt-test", session=session)

    with pytest.raises(ChatClientError):
        client.chat([ChatMessage(role="user", content="hello")], temperature=0.1, max_tokens=16)

    client.close()
    assert session.closed
