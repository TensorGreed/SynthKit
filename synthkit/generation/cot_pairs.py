"""Chain-of-thought generator that expands chunks into reasoning pairs."""

from __future__ import annotations

import json
import logging
from typing import List, Optional, Dict, Any

from .base import BaseGenerator, GeneratedItem
from ..models.client_base import ChatMessage
from ..extensions import register_generator

logger = logging.getLogger(__name__)


class CoTGenerator(BaseGenerator):
    """Generate question/reasoning/answer triples for each text chunk."""

    def build_messages(
        self,
        chunk: str,
        summary: Optional[str],
        num_items: int,
    ) -> List[ChatMessage]:
        """Fill the configured prompt template and return a single user turn."""
        prompt = self.cfg.prompts.cot_generation.format(
            text=chunk,
            summary=summary or "",
            num_pairs=num_items,
        )
        return [ChatMessage(role="user", content=prompt)]

    def parse_output(self, raw: str, chunk_meta: Dict[str, Any]) -> List[GeneratedItem]:
        """Parse JSON output into ``GeneratedItem`` records."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # The generator prompt contract states it returns JSON; drop chunk if not.
            logger.warning("Failed to parse CoT generator output for %s", chunk_meta)
            return []

        items: List[GeneratedItem] = []
        for idx, datum in enumerate(data):
            items.append(
                GeneratedItem(
                    kind="cot",
                    payload={
                        "question": datum.get("question", "").strip(),
                        "reasoning": datum.get("reasoning", "").strip(),
                        "answer": datum.get("answer", "").strip(),
                    },
                    meta={
                        **chunk_meta,
                        "index": idx,
                        "source": "cot_generation",
                    },
                )
            )
        return items


register_generator("cot", lambda client, cfg: CoTGenerator(client, cfg))
