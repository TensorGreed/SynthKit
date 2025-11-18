"""Document discovery helpers used during the harvest stage."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Iterable

from .txt_reader import read_txt
from .pdf_reader import read_pdf

logger = logging.getLogger(__name__)


@dataclass
class HarvestedDoc:
    """Normalized representation of a harvested document."""

    source_path: Path
    text: str


def discover_source_files(root: Path) -> List[Path]:
    """Return files under ``root`` with supported extensions."""
    # Extend this set when additional ingestion formats are supported.
    exts = {".txt", ".pdf"}
    files = [path for path in root.rglob("*") if path.suffix.lower() in exts]
    logger.debug("Discovered %d harvestable files under %s", len(files), root)
    return files


def load_and_normalize(path: Path) -> HarvestedDoc:
    """Read supported files and emit ``HarvestedDoc`` objects."""
    ext = path.suffix.lower()
    if ext == ".txt":
        logger.debug("Reading text document: %s", path)
        text = read_txt(path)
    elif ext == ".pdf":
        logger.debug("Reading PDF document: %s", path)
        text = read_pdf(path)
    else:
        raise ValueError(f"Unsupported extension: {ext}")
    return HarvestedDoc(source_path=path, text=text)


def iter_harvested(root: Path) -> Iterable[HarvestedDoc]:
    """Yield harvested documents lazily so downstream consumers can stream."""
    for file_path in discover_source_files(root):
        logger.debug("Loading harvested document from %s", file_path)
        yield load_and_normalize(file_path)
