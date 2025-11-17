"""Package stage that reformats curated samples into export schemas."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, List

from ..config import ForgeConfig
from ..export.writers import reformat_and_write


def run_package(
    cfg: ForgeConfig,
    fmt: Literal["alpaca", "chatml", "openai-ft"] = "alpaca",
) -> List[Path]:
    """Reformat audited files into JSONL files for downstream consumption."""
    packaged_dir = cfg.io.packaged_path
    packaged_dir.mkdir(parents=True, exist_ok=True)

    outputs: List[Path] = []
    for audited_file in cfg.io.audited_path.glob("*.audited.json"):
        samples = json.loads(audited_file.read_text(encoding="utf-8"))
        out_path = packaged_dir / audited_file.name.replace(".audited.json", f".{fmt}.jsonl")
        reformat_and_write(samples, fmt=fmt, out_path=out_path)
        outputs.append(out_path)
    return outputs
