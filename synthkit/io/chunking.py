from __future__ import annotations
from typing import List


def chunk_text(
    text: str,
    chunk_size: int,
    overlap: int,
) -> List[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    L = len(text)
    while start < L:
        end = min(L, start + chunk_size)
        chunks.append(text[start:end])
        if end == L:
            break
        start = max(0, end - overlap)
    return chunks
