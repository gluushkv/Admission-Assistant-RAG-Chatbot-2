from __future__ import annotations

import gc
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import torch

from experiments.generation.contexts import (
    build_retrieved_contexts,
    save_contexts,
)
from experiments.generation.run_generator_exp import (
    load_generation_config,
    run_generation_model,
    save_raw_records,
)
from rag.evaluation.models import GoldenQuestion
from rag.generation.factory import create_generator
from rag.generation.generator import GenerationBackend
from rag.models import Document
from rag.retrieval.qdrant_store import QdrantStore


def run_retrieved_experiment(
    *,
    documents: Sequence[Document],
    questions: Sequence[GoldenQuestion],
    store: QdrantStore,
    config_path: str | Path = "configs/generation.yaml",
    output_dir: str | Path = "results/generation",
    device: str = "cuda",
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> list[dict]:

    if not documents:
        raise ValueError("documents must not be empty")

    if not questions:
        raise ValueError("questions must not be empty")

    config = load_generation_config(config_path)

    common = config["common"]
    model_configs = config["models"]
    retrieval_config = config.get(
        "retrieved_context",
        {},
    )

    output_dir = Path(output_dir)
    raw_dir = output_dir / "raw"
    contexts_dir = output_dir / "contexts"

    print(
        "Preparing Retrieved Context: "
        "Markdown -> BGE-M3 -> top-10 -> "
        "BGE reranker -> top-5"
    )

    retrieved_contexts = build_retrieved_contexts(
        documents=documents,
        questions=questions,
        store=store,
        candidate_k=int(
            retrieval_config.get(
                "candidate_k",
                10,
            )
        ),
        final_k=int(
            retrieval_config.get(
                "final_k",
                5,
            )
        ),
        embedding_model_name=str(
            retrieval_config.get(
                "embedding_model",
                "bge-m3",
            )
        ),
        embedding_batch_size=int(
            retrieval_config.get(
                "embedding_batch_size",
                32,
            )
        ),
        reranker_model_id=str(
            retrieval_config.get(
                "reranker_model_id",
                "BAAI/bge-reranker-v2-m3",
            )
        ),
        reranker_batch_size=int(
            retrieval_config.get(
                "reranker_batch_size",
                16,
            )
        ),
        collection_name=str(
            retrieval_config.get(
                "collection_name",
                "generation-retrieved-context",
            )
        ),
        device=device,
    )

    save_contexts(
        contexts=retrieved_contexts,
        output_path=(
            contexts_dir
            / "retrieved_contexts.jsonl"
        ),
    )

    all_records: list[dict] = []

    for configuration, model_config in model_configs.items():

        print()
        print("=" * 40)
        print(
            f"Generation model: {configuration}"
        )
        print("=" * 40)

        generator = create_generator(
            model_id=str(
                model_config["model_id"]
            ),
            backend=cast(
                GenerationBackend,
                model_config["backend"],
            ),
            device_map=device_map,
            torch_dtype=torch_dtype,
            max_new_tokens=int(
                model_config.get(
                    "max_new_tokens",
                    common.get(
                        "max_new_tokens",
                        256,
                    ),
                )
            ),
            do_sample=bool(
                model_config.get(
                    "do_sample",
                    common.get(
                        "do_sample",
                        False,
                    ),
                )
            ),
            temperature=model_config.get(
                "temperature",
                common.get("temperature"),
            ),
            top_p=model_config.get(
                "top_p",
                common.get("top_p"),
            ),
            enable_thinking=model_config.get(
                "enable_thinking"
            ),
            trust_remote_code=bool(
                model_config.get(
                    "trust_remote_code",
                    False,
                )
            ),
        )

        try:
            retrieved_records = run_generation_model(
                generator=generator,
                questions=questions,
                contexts=retrieved_contexts,
                configuration=configuration,
                context_type="retrieved",
                warmup_queries=int(
                    common.get(
                        "warmup_queries",
                        3,
                    )
                ),
            )

            save_raw_records(
                records=retrieved_records,
                output_path=(
                    raw_dir
                    / (
                        f"{configuration}_"
                        "retrieved.jsonl"
                    )
                ),
            )

            all_records.extend(
                retrieved_records
            )

        finally:
            del generator
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return all_records