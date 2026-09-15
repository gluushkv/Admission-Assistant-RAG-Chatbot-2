from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast

from experiments.chunking.models import (
    ChunkingConfiguration,
    ConfigurationResult,
)
from experiments.chunking.runner import (
    load_chunking_config,
    run_chunking_configuration,
    save_experiment_results,
)
from rag.chunking.factory import ChunkingStrategy
from rag.embeddings.embedder import SentenceTransformerEmbedder
from rag.evaluation.models import GoldenQuestion
from rag.models import Document
from rag.retrieval.qdrant_store import QdrantStore


def run_method_experiment(
    *,
    documents: Sequence[Document],
    questions: Sequence[GoldenQuestion],
    embedder: SentenceTransformerEmbedder,
    store: QdrantStore,
    config_path: str | Path = "configs/chunking.yaml",
    output_dir: str | Path = "results/chunking/method",
) -> list[ConfigurationResult]:

    config = load_chunking_config(
        config_path
    )

    common = config["common"]
    experiment = config["method_experiment"]


    chunk_size = int(
        experiment["chunk_size"]
    )

    chunk_overlap = int(
        experiment["chunk_overlap"]
    )

    configurations = [
        ChunkingConfiguration(
            name=strategy_name,
            strategy=cast(
                ChunkingStrategy,
                strategy_name,
            ),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        for strategy_name
        in experiment["strategies"]
    ]

    results: list[ConfigurationResult] = []

    for configuration in configurations:
        print(
            f"Running chunking strategy: "
            f"{configuration.strategy}"
        )

        result = run_chunking_configuration(
            documents=documents,
            questions=questions,
            configuration=configuration,
            embedder=embedder,
            store=store,
            ks=tuple(common["ks"]),
            sentence_language=common.get(
                "sentence_language",
                "russian",
            ),
            collection_name=(
                "chunking-method-experiment"
            ),
        )

        results.append(result)

    save_experiment_results(
        results=results,
        output_dir=output_dir,
    )

    return results