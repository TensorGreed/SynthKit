from __future__ import annotations
from pathlib import Path
from typing import List

from ..config import ForgeConfig
from ..io.loaders import iter_harvested


def run_harvest(cfg: ForgeConfig) -> List[Path]:
    out_dir = cfg.io.harvested_path
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    for doc in iter_harvested(cfg.io.input_root):
        rel = doc.source_path.relative_to(cfg.io.input_root)
        out_path = out_dir / (rel.as_posix().replace("/", "__") + ".txt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(doc.text, encoding="utf-8")
        written.append(out_path)

    return written
