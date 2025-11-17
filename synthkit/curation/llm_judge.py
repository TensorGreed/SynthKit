"""LLM-powered judge that scores generated samples."""

from __future__ import annotations

import json
from typing import Dict, Any

from .judge_base import JudgedItem
from ..config import ForgeConfig
from ..models.client_base import ChatClient, ChatMessage


class LLMJudge:
    """Query an LLM with a rating prompt and normalize the response."""

    def __init__(self, client: ChatClient, cfg: ForgeConfig):
        self.client = client
        self.cfg = cfg

    def _build_prompt(self, sample: Dict[str, Any]) -> str:
        """Render the rating template using the common QA fields."""
        # Expect at least question/answer keys for QA, but keep things generic so
        # other shapes (response, completion, etc.) still map cleanly.
        question = sample.get("question", "")
        answer = sample.get("answer", sample.get("response", ""))
        tmpl = self.cfg.prompts.qa_rating
        return tmpl.format(question=question, answer=answer)

    def judge(self, sample: Dict[str, Any]) -> JudgedItem:
        """Score a sample and convert the JSON payload into ``JudgedItem``."""
        prompt = self._build_prompt(sample)
        messages = [ChatMessage(role="user", content=prompt)]
        raw = self.client.chat(
            messages,
            temperature=0.0,
            max_tokens=self.cfg.curation.max_tokens,
        )
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # A malformed response should fail closed to avoid leaking low-quality data.
            return JudgedItem(
                score=0.0,
                keep=False,
                label="parse_error",
                rationale="Could not parse judge JSON",
                original=sample,
            )

        score = float(data.get("score", 0.0))
        label = data.get("label", "ok" if score >= self.cfg.curation.min_score else "bad")
        rationale = data.get("reason", "")
        keep = score >= self.cfg.curation.min_score

        return JudgedItem(
            score=score,
            keep=keep,
            label=label,
            rationale=rationale,
            original=sample,
        )
