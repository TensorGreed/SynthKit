from __future__ import annotations
import json
from pathlib import Path
from typing import List

from ..config import ForgeConfig
from ..models.router import ModelRouter
from ..curation.llm_judge import LLMJudge


def run_audit(cfg: ForgeConfig) -> List[Path]:
    router = ModelRouter(cfg)
    judge_client = router.for_stage(cfg.models.audit_judge)
    judge = LLMJudge(judge_client, cfg)

    audited_dir = cfg.io.audited_path
    audited_dir.mkdir(parents=True, exist_ok=True)

    outputs: List[Path] = []
    for minted_file in cfg.io.minted_path.glob("*.json"):
        raw = json.loads(minted_file.read_text(encoding="utf-8"))
        curated = []
        for ex in raw:
            judged = judge.judge(ex)
            if judged.keep:
                curated.append(ex | {
                    "score": judged.score,
                    "label": judged.label,
                    "judge_rationale": judged.rationale,
                })

        out_path = audited_dir / minted_file.name.replace(".json", ".audited.json")
        out_path.write_text(json.dumps(curated, ensure_ascii=False, indent=2), encoding="utf-8")
        outputs.append(out_path)

    return outputs
