import pytest

from synthkit import extensions


def test_register_and_get_generator(monkeypatch):
    monkeypatch.setattr(extensions, "_generator_registry", {})

    def factory(client, cfg):  # pragma: no cover - dummy callable
        return None

    extensions.register_generator("custom", factory)
    retrieved = extensions.get_generator_factory("custom")
    assert retrieved is factory
    assert "custom" in extensions.available_generator_types()

    with pytest.raises(ValueError):
        extensions.register_generator("custom", factory)


def test_register_and_get_formatter(monkeypatch):
    monkeypatch.setattr(extensions, "_formatter_registry", {})

    def formatter(sample):  # pragma: no cover - dummy callable
        return sample

    extensions.register_formatter("demo", formatter)
    assert extensions.get_formatter("demo") is formatter
    assert "demo" in extensions.available_formatter_names()

    with pytest.raises(ValueError):
        extensions.register_formatter("demo", formatter)

    with pytest.raises(KeyError):
        extensions.get_formatter("missing")
