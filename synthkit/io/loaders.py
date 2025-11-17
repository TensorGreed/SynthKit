from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Iterable

from .txt_reader import read_txt
from .pdf_reader import read_pdf


@dataclass
class HarvestedDoc:
    source_path: Path
    text: str


def discover_source_files(root: Path) -> List[Path]:
    # Add more extensions as needed
    exts = {".txt", ".pdf"}
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]


def load_and_normalize(path: Path) -> HarvestedDoc:
    ext = path.suffix.lower()
    if ext == ".txt":
        txt = read_txt(path)
    elif ext == ".pdf":
        txt = read_pdf(path)
    else:
        raise ValueError(f"Unsupported extension: {ext}")
    return HarvestedDoc(source_path=path, text=txt)


def iter_harvested(root: Path) -> Iterable[HarvestedDoc]:
    for f in discover_source_files(root):
        yield load_and_normalize(f)
