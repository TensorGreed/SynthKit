from __future__ import annotations
import json
from typing import Dict, Any

from .judge_base import JudgedItem
from ..config import ForgeConfig
from ..models.client_base import ChatClient, ChatMessage


class LLMJudge:
    def __init__(self, client: ChatClient, cfg: ForgeConfig):
        self.client = client
        self.cfg = cfg

    def _build_prompt(self, sample: Dict[str, Any]) -> str:
        # Expect at least question/answer keys for QA, but generic enough
        question = sample.get("question", "")
        answer = sample.get("answer", sample.get("response", ""))
        tmpl = self.cfg.prompts.qa_rating
        return tmpl.format(question=question, answer=answer)

    def judge(self, sample: Dict[str, Any]) -> JudgedItem:
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
            # Fallback: reject
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
