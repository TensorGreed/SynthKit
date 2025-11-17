from __future__ import annotations
from typing import Dict, Any


def to_alpaca(sample: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "instruction": sample.get("question", ""),
        "input": "",
        "output": sample.get("answer", ""),
    }


def to_chatml(sample: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": sample.get("question", "")},
            {"role": "assistant", "content": sample.get("answer", "")},
        ]
    }


def to_openai_ft(sample: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": sample.get("question", "")},
            {"role": "assistant", "content": sample.get("answer", "")},
        ]
    }


FORMATTERS = {
    "alpaca": to_alpaca,
    "chatml": to_chatml,
    "openai-ft": to_openai_ft,
}
