from __future__ import annotations

import gc
from collections.abc import Sequence
from pathlib import Path

import torch

from experiments.generation.contexts import (
    build_gold_contexts,
    build_retrieved_contexts,
)
from experiments.generation.run_generator_exp import (
    create_generator_for_configuration,
    load_generation_config,
    run_generation_model,
    save_raw_records,
)
from rag.evaluation.models import (
    GoldenQuestion,
)
from rag.models import Document
from rag.retrieval.qdrant_store import (
    QdrantStore,
)


def run_gold_vs_retrieved_experiment(
    *,
    documents: Sequence[Document],
    questions: Sequence[
        GoldenQuestion
    ],
    store: QdrantStore,
    config_path: str | Path = (
        "configs/generation.yaml"
    ),
    output_dir: str | Path = (
        "results/generation/raw"
    ),
    device: str = "cuda",
    device_map: str = "auto",
    torch_dtype: torch.dtype = (
        torch.bfloat16
    ),
) -> None:
    output_dir = Path(
        output_dir
    )

    config = load_generation_config(
        config_path
    )

    print(
        "Building Gold Context..."
    )

    gold_contexts = (
        build_gold_contexts(
            questions
        )
    )

    print(
        "Building Retrieved Context..."
    )


    retrieved_contexts = (
        build_retrieved_contexts(
            documents=documents,
            questions=questions,
            store=store,
            device=device,
            candidate_k=10,
            final_k=5,
        )
    )


    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    warmup_queries = int(
        config["common"].get(
            "warmup_queries",
            3,
        )
    )

    for configuration in config[
        "models"
    ]:
        print()
        print(
            "================================="
        )
        print(
            f"Generation model: "
            f"{configuration}"
        )
        print(
            "================================="
        )

        generator = (
            create_generator_for_configuration(
                configuration=configuration,
                config=config,
                device_map=device_map,
                torch_dtype=torch_dtype,
            )
        )

        try:
            print(
                "Running Gold Context..."
            )

            gold_records = (
                run_generation_model(
                    generator=generator,
                    questions=questions,
                    contexts=gold_contexts,
                    configuration=(
                        configuration
                    ),
                    context_type="gold",
                    warmup_queries=(
                        warmup_queries
                    ),
                )
            )

            save_raw_records(
                records=gold_records,
                output_path=(
                    output_dir
                    / (
                        f"{configuration}"
                        "_gold.jsonl"
                    )
                ),
            )

            print(
                "Running Retrieved Context..."
            )

            retrieved_records = (
                run_generation_model(
                    generator=generator,
                    questions=questions,
                    contexts=(
                        retrieved_contexts
                    ),
                    configuration=(
                        configuration
                    ),
                    context_type=(
                        "retrieved"
                    ),
                    warmup_queries=0,
                )
            )

            save_raw_records(
                records=retrieved_records,
                output_path=(
                    output_dir
                    / (
                        f"{configuration}"
                        "_retrieved.jsonl"
                    )
                ),
            )

        finally:
            del generator

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()