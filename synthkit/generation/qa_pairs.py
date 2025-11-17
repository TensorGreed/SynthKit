from __future__ import annotations
import json
from typing import List, Optional, Dict, Any

from .base import BaseGenerator, GeneratedItem
from ..models.client_base import ChatMessage


class QAGenerator(BaseGenerator):
    def build_messages(
        self,
        chunk: str,
        summary: Optional[str],
        num_items: int,
    ) -> List[ChatMessage]:
        prompt = self.cfg.prompts.qa_generation.format(
            text=chunk,
            summary=summary or "",
            num_pairs=num_items,
        )
        return [ChatMessage(role="user", content=prompt)]

    def parse_output(self, raw: str, chunk_meta: Dict[str, Any]) -> List[GeneratedItem]:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # You can add recovery heuristics here
            return []

        items: List[GeneratedItem] = []
        for idx, d in enumerate(data):
            items.append(
                GeneratedItem(
                    kind="qa",
                    payload={
                        "question": d.get("question", "").strip(),
                        "answer": d.get("answer", "").strip(),
                    },
                    meta={
                        **chunk_meta,
                        "index": idx,
                        "source": "qa_generation",
                    },
                )
            )
        return items
