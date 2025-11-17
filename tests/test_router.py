from pathlib import Path

from synthkit.config import (
    ForgeConfig,
    IOSettings,
    StageModels,
    ModelRef,
    PromptSet,
    GenerationSettings,
    CurationSettings,
    ProviderConfig,
)
from synthkit.models import router as router_module


class DummyClient:
    def __init__(self):
        self.closed = False

    def close(self):  # pragma: no cover - invoked by router
        self.closed = True


def _build_cfg(tmp_path: Path) -> ForgeConfig:
    provider = ProviderConfig(type="http", api_base="https://example.com")
    model_ref = ModelRef(provider="default", name="dummy")
    return ForgeConfig(
        io=IOSettings(
            input_root=tmp_path,
            working_root=tmp_path,
        ),
        models=StageModels(
            harvest_summarizer=model_ref,
            mint_generator=model_ref,
            audit_judge=model_ref,
            package_validator=model_ref,
        ),
        prompts=PromptSet(
            qa_generation="",
            cot_generation="",
            qa_rating="",
        ),
        generation=GenerationSettings(),
        curation=CurationSettings(),
        providers={"default": provider},
    )


def test_model_router_caches_and_closes(monkeypatch, tmp_path):
    cfg = _build_cfg(tmp_path)
    created_clients: list[DummyClient] = []

    def fake_builder(provider_cfg, model_name):  # pragma: no cover - patched behavior
        client = DummyClient()
        created_clients.append(client)
        return client

    monkeypatch.setattr(router_module, "_build_client", fake_builder)

    router = router_module.ModelRouter(cfg)
    ref = cfg.models.mint_generator

    client_a = router.for_stage(ref)
    client_b = router.for_stage(ref)

    assert client_a is client_b
    assert len(created_clients) == 1

    router.close_all()
    assert created_clients[0].closed
