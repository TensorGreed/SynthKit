from __future__ import annotations
import json
from pathlib import Path
from typing import Literal, List

from ..config import ForgeConfig
from ..models.router import ModelRouter
from ..io.chunking import chunk_text
from ..generation.qa_pairs import QAGenerator
from ..generation.cot_pairs import CoTGenerator
from ..models.client_base import ChatMessage
from ..models.client_base import ChatClient


def _build_summarizer(router: ModelRouter, cfg: ForgeConfig) -> ChatClient:
    ref = cfg.models.harvest_summarizer
    return router.for_stage(ref)


def _summarize_chunk(
    client: ChatClient,
    chunk: str,
    max_tokens: int,
) -> str:
    prompt = (
        "Summarize the following text in 3–5 sentences, preserving key technical details:\n\n"
        + chunk
    )
    msgs = [ChatMessage(role="user", content=prompt)]
    return client.chat(msgs, temperature=0.2, max_tokens=max_tokens)


def run_mint(
    cfg: ForgeConfig,
    generator_type: Literal["qa", "cot"] = "qa",
) -> List[Path]:
    router = ModelRouter(cfg)
    gen_client = router.for_stage(cfg.models.mint_generator)
    summarizer = _build_summarizer(router, cfg)

    if generator_type == "qa":
        generator = QAGenerator(gen_client, cfg)
    else:
        generator = CoTGenerator(gen_client, cfg)

    minted_dir = cfg.io.minted_path
    minted_dir.mkdir(parents=True, exist_ok=True)

    outputs: List[Path] = []
    for txt_file in cfg.io.harvested_path.glob("*.txt"):
        text = txt_file.read_text(encoding="utf-8")
        chunks = chunk_text(
            text,
            chunk_size=cfg.generation.chunk_size,
            overlap=cfg.generation.chunk_overlap,
        )

        all_items = []
        remaining = cfg.generation.max_pairs_per_doc
        for idx, ch in enumerate(chunks):
            if remaining <= 0:
                break
            summary = _summarize_chunk(summarizer, ch, max_tokens=256)
            items = generator.generate(
                chunk=ch,
                summary=summary,
                num_items=min(remaining, 8),
                chunk_meta={"source_file": str(txt_file), "chunk_index": idx},
            )
            all_items.extend(items)
            remaining = cfg.generation.max_pairs_per_doc - len(all_items)

        payload = [i.payload | {"meta": i.meta} for i in all_items]
        out_path = minted_dir / (txt_file.stem + f".{generator_type}.json")
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        outputs.append(out_path)

    return outputs
