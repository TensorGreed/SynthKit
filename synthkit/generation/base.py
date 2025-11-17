from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from ..models.client_base import ChatClient, ChatMessage
from ..config import ForgeConfig


@dataclass
class GeneratedItem:
    kind: str           # "qa" | "cot" | "classifier" | etc.
    payload: Dict[str, Any]
    meta: Dict[str, Any]


class BaseGenerator:
    """Abstract generator that turns a text chunk into structured examples."""

    def __init__(self, client: ChatClient, cfg: ForgeConfig):
        self.client = client
        self.cfg = cfg

    def build_messages(
        self,
        chunk: str,
        summary: Optional[str],
        num_items: int,
    ) -> List[ChatMessage]:
        raise NotImplementedError

    def parse_output(self, raw: str, chunk_meta: Dict[str, Any]) -> List[GeneratedItem]:
        raise NotImplementedError

    def generate(
        self,
        chunk: str,
        summary: Optional[str],
        num_items: int,
        chunk_meta: Dict[str, Any],
    ) -> List[GeneratedItem]:
        messages = self.build_messages(chunk, summary, num_items)
        raw = self.client.chat(
            messages,
            temperature=self.cfg.generation.temperature,
            max_tokens=self.cfg.generation.max_tokens,
        )
        return self.parse_output(raw, chunk_meta)
