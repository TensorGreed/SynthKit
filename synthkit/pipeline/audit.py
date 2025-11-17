"""Audit stage that scores minted samples and filters low-quality data."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, List

from ..config import ForgeConfig
from ..models.router import ModelRouter
from ..curation.llm_judge import LLMJudge

logger = logging.getLogger(__name__)


def _is_valid_sample(sample: Dict[str, Any]) -> bool:
    """Basic schema validation before calling expensive LLM judges."""
    if not isinstance(sample, dict):
        return False
    question = sample.get("question")
    answer = sample.get("answer", sample.get("response"))
    return isinstance(question, str) and isinstance(answer, str)


def run_audit(cfg: ForgeConfig) -> List[Path]:
    """Run the LLM judge across all minted files and save curated outputs."""
    router = ModelRouter(cfg)
    judge_client = router.for_stage(cfg.models.audit_judge)
    judge = LLMJudge(judge_client, cfg)

    audited_dir = cfg.io.audited_path
    audited_dir.mkdir(parents=True, exist_ok=True)

    outputs: List[Path] = []
    try:
        for minted_file in cfg.io.minted_path.glob("*.json"):
            raw = json.loads(minted_file.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                logger.warning("Skipping %s; expected list payload", minted_file.name)
                continue
            curated = []
            for sample in raw:
                if not _is_valid_sample(sample):
                    logger.warning("Dropping malformed sample from %s", minted_file.name)
                    continue
                judged = judge.judge(sample)
                if judged.keep:
                    curated.append(
                        sample
                        | {
                            "score": judged.score,
                            "label": judged.label,
                            "judge_rationale": judged.rationale,
                        }
                    )

            out_path = audited_dir / minted_file.name.replace(".json", ".audited.json")
            # Persist curated payloads with deterministic formatting for diff-friendly review.
            out_path.write_text(
                json.dumps(curated, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            outputs.append(out_path)
    finally:
        router.close_all()

    return outputs
