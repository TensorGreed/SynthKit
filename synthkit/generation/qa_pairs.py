"""QA pair generator that produces question/answer outputs per chunk."""

from __future__ import annotations

import json
from typing import List, Optional, Dict, Any

from .base import BaseGenerator, GeneratedItem
from ..models.client_base import ChatMessage


class QAGenerator(BaseGenerator):
    """Generate question/answer pairs from harvested chunks."""

    def build_messages(
        self,
        chunk: str,
        summary: Optional[str],
        num_items: int,
    ) -> List[ChatMessage]:
        """Fill the QA prompt template with the source chunk."""
        prompt = self.cfg.prompts.qa_generation.format(
            text=chunk,
            summary=summary or "",
            num_pairs=num_items,
        )
        return [ChatMessage(role="user", content=prompt)]

    def parse_output(self, raw: str, chunk_meta: Dict[str, Any]) -> List[GeneratedItem]:
        """Parse JSON output into structured QA records."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # The downstream pipeline expects JSON arrays; skip malformed responses.
            return []

        items: List[GeneratedItem] = []
        for idx, datum in enumerate(data):
            items.append(
                GeneratedItem(
                    kind="qa",
                    payload={
                        "question": datum.get("question", "").strip(),
                        "answer": datum.get("answer", "").strip(),
                    },
                    meta={
                        **chunk_meta,
                        "index": idx,
                        "source": "qa_generation",
                    },
                )
            )
        return items
