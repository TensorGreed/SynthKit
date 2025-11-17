"""Plain-text reader used during harvesting."""

from pathlib import Path


def read_txt(path: Path) -> str:
    """Return UTF-8 text while ignoring decode errors."""
    return path.read_text(encoding="utf-8", errors="ignore")
