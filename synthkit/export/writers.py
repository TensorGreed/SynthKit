from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable, Dict, Any

from .formats import FORMATTERS


def write_jsonl(
    samples: Iterable[Dict[str, Any]],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def reformat_and_write(
    samples: Iterable[Dict[str, Any]],
    fmt: str,
    out_path: Path,
) -> None:
    formatter = FORMATTERS[fmt]
    formatted = (formatter(s) for s in samples)
    write_jsonl(formatted, out_path)
