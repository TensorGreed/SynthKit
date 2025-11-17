from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional

import yaml


@dataclass
class ModelRef:
    """Reference to a logical model used in a stage."""
    provider: str          # "openai" | "anthropic" | "http"
    name: str              # model name for that provider
    profile: str | None = None  # optional profile name (e.g. "fast", "strong")


@dataclass
class StageModels:
    """Models used at each stage."""
    harvest_summarizer: ModelRef
    mint_generator: ModelRef
    audit_judge: ModelRef
    package_validator: ModelRef


@dataclass
class PromptSet:
    """Prompt templates for various tasks."""
    qa_generation: str
    cot_generation: str
    qa_rating: str
    classifier_generation: str | None = None


@dataclass
class GenerationSettings:
    temperature: float = 0.7
    max_tokens: int = 1024
    chunk_size: int = 4000
    chunk_overlap: int = 200
    max_pairs_per_doc: int = 50


@dataclass
class CurationSettings:
    min_score: float = 7.0
    max_tokens: int = 512


@dataclass
class IOSettings:
    input_root: Path
    working_root: Path
    harvested_dir: str = "harvested"
    minted_dir: str = "minted"
    audited_dir: str = "audited"
    packaged_dir: str = "packaged"

    @property
    def harvested_path(self) -> Path:
        return self.working_root / self.harvested_dir

    @property
    def minted_path(self) -> Path:
        return self.working_root / self.minted_dir

    @property
    def audited_path(self) -> Path:
        return self.working_root / self.audited_dir

    @property
    def packaged_path(self) -> Path:
        return self.working_root / self.packaged_dir


@dataclass
class ProviderConfig:
    """Generic provider configuration (API base, keys, etc.)."""
    type: str           # "openai", "anthropic", "http"
    api_base: str
    api_key_env: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ForgeConfig:
    io: IOSettings
    models: StageModels
    prompts: PromptSet
    generation: GenerationSettings
    curation: CurationSettings
    providers: Dict[str, ProviderConfig]


def _load_model_ref(raw: Dict[str, Any]) -> ModelRef:
    return ModelRef(
        provider=raw["provider"],
        name=raw["name"],
        profile=raw.get("profile"),
    )


def _load_stage_models(raw: Dict[str, Any]) -> StageModels:
    return StageModels(
        harvest_summarizer=_load_model_ref(raw["harvest_summarizer"]),
        mint_generator=_load_model_ref(raw["mint_generator"]),
        audit_judge=_load_model_ref(raw["audit_judge"]),
        package_validator=_load_model_ref(raw["package_validator"]),
    )


def _load_providers(raw: Dict[str, Any]) -> Dict[str, ProviderConfig]:
    providers: Dict[str, ProviderConfig] = {}
    for key, cfg in raw.items():
        providers[key] = ProviderConfig(
            type=cfg["type"],
            api_base=cfg["api_base"],
            api_key_env=cfg.get("api_key_env"),
            extra=cfg.get("extra", {}),
        )
    return providers


def load_config(path: str | Path) -> ForgeConfig:
    data = yaml.safe_load(Path(path).read_text())

    io = IOSettings(
        input_root=Path(data["io"]["input_root"]).expanduser(),
        working_root=Path(data["io"]["working_root"]).expanduser(),
        harvested_dir=data["io"].get("harvested_dir", "harvested"),
        minted_dir=data["io"].get("minted_dir", "minted"),
        audited_dir=data["io"].get("audited_dir", "audited"),
        packaged_dir=data["io"].get("packaged_dir", "packaged"),
    )

    prompts = PromptSet(
        qa_generation=data["prompts"]["qa_generation"],
        cot_generation=data["prompts"]["cot_generation"],
        qa_rating=data["prompts"]["qa_rating"],
        classifier_generation=data["prompts"].get("classifier_generation"),
    )

    gen = GenerationSettings(**data.get("generation", {}))
    cur = CurationSettings(**data.get("curation", {}))
    models = _load_stage_models(data["models"])
    providers = _load_providers(data["providers"])

    return ForgeConfig(
        io=io,
        models=models,
        prompts=prompts,
        generation=gen,
        curation=cur,
        providers=providers,
    )
