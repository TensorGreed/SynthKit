"""QA pair generator that produces question/answer outputs per chunk."""

from __future__ import annotations

import json
import logging
from typing import List, Optional, Dict, Any

from .base import BaseGenerator, GeneratedItem
from ..models.client_base import ChatMessage
from ..extensions import register_generator

logger = logging.getLogger(__name__)


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
            preview = raw.strip().replace("\n", " ")
            if len(preview) > 400:
                preview = preview[:400] + "..."
            logger.warning(
                "Failed to parse QA generator output for %s. Raw preview: %s",
                chunk_meta,
                preview or "<empty response>",
            )
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


register_generator("qa", lambda client, cfg: QAGenerator(client, cfg))
