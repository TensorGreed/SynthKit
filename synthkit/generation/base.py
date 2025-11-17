"""Abstract generator definitions for turning text chunks into samples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from ..models.client_base import ChatClient, ChatMessage
from ..config import ForgeConfig


@dataclass
class GeneratedItem:
    """Synthetic datum produced by a generator with accompanying metadata."""

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
        """Return the chat payload that asks the LLM to produce ``num_items``."""
        raise NotImplementedError

    def parse_output(self, raw: str, chunk_meta: Dict[str, Any]) -> List[GeneratedItem]:
        """Convert raw model output into ``GeneratedItem`` records."""
        raise NotImplementedError

    def generate(
        self,
        chunk: str,
        summary: Optional[str],
        num_items: int,
        chunk_meta: Dict[str, Any],
    ) -> List[GeneratedItem]:
        """Call ``chat`` then parse the response into structured samples."""
        messages = self.build_messages(chunk, summary, num_items)
        raw = self.client.chat(
            messages,
            temperature=self.cfg.generation.temperature,
            max_tokens=self.cfg.generation.max_tokens,
        )
        return self.parse_output(raw, chunk_meta)
