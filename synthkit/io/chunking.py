"""Utility helpers for chunking normalized documents."""

from __future__ import annotations

from typing import List


def chunk_text(
    text: str,
    chunk_size: int,
    overlap: int,
) -> List[str]:
    """Return sliding-window chunks with ``overlap`` characters between slices."""
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    total_len = len(text)
    while start < total_len:
        end = min(total_len, start + chunk_size)
        chunks.append(text[start:end])
        if end == total_len:
            break
        # Step forward while rewinding ``overlap`` chars to preserve context.
        start = max(0, end - overlap)
    return chunks
