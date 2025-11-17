"""Convenience helper that executes the full pipeline sequentially."""

from __future__ import annotations

from typing import Literal

from ..config import ForgeConfig
from .harvest import run_harvest
from .mint import run_mint
from .audit import run_audit
from .package import run_package


def run_pipeline(
    cfg: ForgeConfig,
    generator_type: Literal["qa", "cot"] = "qa",
    export_fmt: Literal["alpaca", "chatml", "openai-ft"] = "alpaca",
) -> None:
    """Execute each stage in order, surfacing progress on stdout."""
    print("-> Stage 1: harvest")
    run_harvest(cfg)

    print("-> Stage 2: mint")
    run_mint(cfg, generator_type=generator_type)

    print("-> Stage 3: audit")
    run_audit(cfg)

    print("-> Stage 4: package")
    run_package(cfg, fmt=export_fmt)
