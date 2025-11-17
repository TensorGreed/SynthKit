"""Writers that persist curated samples in a streaming-friendly JSONL format."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Dict, Any

from ..extensions import get_formatter


def write_jsonl(
    samples: Iterable[Dict[str, Any]],
    out_path: Path,
) -> None:
    """Write iterable samples to ``out_path`` as JSON lines."""
    # Ensure parent directories exist before streaming out the file.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def reformat_and_write(
    samples: Iterable[Dict[str, Any]],
    fmt: str,
    out_path: Path,
) -> None:
    """Convert records to the requested format then persist as JSONL."""
    try:
        formatter = get_formatter(fmt)
    except KeyError as exc:
        raise ValueError(str(exc)) from exc
    formatted = (formatter(sample) for sample in samples)
    write_jsonl(formatted, out_path)
