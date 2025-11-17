import os

import pytest
import requests

from synthkit.config import ProviderConfig
from synthkit.models.client_base import ChatMessage, ChatClientError
from synthkit.models.openai_client import OpenAIChatClient
from synthkit.models.ollama_client import OllamaChatClient


class DummySession:
    def __init__(self, exception: Exception):
        self._exception = exception
        self.closed = False

    def post(self, *args, **kwargs):  # pragma: no cover - invoked in tests
        raise self._exception

    def close(self):  # pragma: no cover - invoked in tests
        self.closed = True


class RecordingSession:
    def __init__(self, payload):
        self._payload = payload
        self.last_request = None
        self.closed = False

    def post(self, url, json=None, **kwargs):  # pragma: no cover - invoked in tests
        self.last_request = (url, json, kwargs)
        return DummyResponse(self._payload)

    def close(self):  # pragma: no cover - invoked in tests
        self.closed = True


class DummyResponse:
    def __init__(self, payload, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)

    def json(self):
        return self._payload


def test_openai_client_raises_chat_client_error(monkeypatch):
    provider = ProviderConfig(type="openai", api_base="https://example.com")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    session = DummySession(requests.exceptions.RequestException("boom"))
    client = OpenAIChatClient(provider, "gpt-test", session=session)

    with pytest.raises(ChatClientError):
        client.chat([ChatMessage(role="user", content="hello")], temperature=0.1, max_tokens=16)

    client.close()
    assert session.closed


def test_ollama_client_success():
    provider = ProviderConfig(type="ollama", api_base="http://localhost:11434")
    payload = {"message": {"role": "assistant", "content": "Hi there"}}
    session = RecordingSession(payload)
    client = OllamaChatClient(provider, "mistral", session=session)

    msg = [ChatMessage(role="user", content="hello?")]
    result = client.chat(msg, temperature=0.3, max_tokens=42)
    assert result == "Hi there"

    url, sent_payload, kwargs = session.last_request
    assert url.endswith("/api/chat")
    assert sent_payload["options"]["num_predict"] == 42
    assert sent_payload["messages"][0]["content"] == "hello?"

    client.close()
    assert session.closed
