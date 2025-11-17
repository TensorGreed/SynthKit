"""Helpers that normalize curated QA items into export-ready schemas."""

from __future__ import annotations

from typing import Dict, Any


def to_alpaca(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Return a single Alpaca-style record."""
    return {
        "instruction": sample.get("question", ""),
        "input": "",
        "output": sample.get("answer", ""),
    }


def to_chatml(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Return a ChatML payload with user/assistant turns."""
    return {
        "messages": [
            {"role": "user", "content": sample.get("question", "")},
            {"role": "assistant", "content": sample.get("answer", "")},
        ]
    }


def to_openai_ft(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Return an OpenAI fine-tuning JSONL record."""
    return {
        "messages": [
            {"role": "user", "content": sample.get("question", "")},
            {"role": "assistant", "content": sample.get("answer", "")},
        ]
    }


# Mapping of exporter slug -> callable transformer.
FORMATTERS = {
    "alpaca": to_alpaca,
    "chatml": to_chatml,
    "openai-ft": to_openai_ft,
}
