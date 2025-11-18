"""Harvest stage that normalizes raw documents into UTF-8 text."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from ..config import ForgeConfig
from ..io.loaders import iter_harvested

logger = logging.getLogger(__name__)


def run_harvest(cfg: ForgeConfig) -> List[Path]:
    """Normalize supported source files and store them under ``harvested_path``."""
    out_dir = cfg.io.harvested_path
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    logger.info(
        "Harvesting documents from %s into %s",
        cfg.io.input_root,
        out_dir,
    )

    for doc in iter_harvested(cfg.io.input_root):
        rel = doc.source_path.relative_to(cfg.io.input_root)
        # Flatten nested directories into filename-safe tokens for reproducibility.
        out_path = out_dir / (rel.as_posix().replace("/", "__") + ".txt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(doc.text, encoding="utf-8")
        written.append(out_path)
        logger.debug("Wrote harvested copy: %s", out_path)

    if written:
        logger.info("Harvested %d documents.", len(written))
    else:
        logger.warning("No harvestable documents found under %s", cfg.io.input_root)
    return written
