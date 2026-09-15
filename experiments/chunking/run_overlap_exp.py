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


def run_overlap_experiment(
    *,
    documents: Sequence[Document],
    questions: Sequence[GoldenQuestion],
    embedder: SentenceTransformerEmbedder,
    store: QdrantStore,
    config_path: str | Path = "configs/chunking.yaml",
    output_dir: str | Path = "results/chunking/overlap",
) -> list[ConfigurationResult]:

    config = load_chunking_config(
        config_path
    )

    common = config["common"]
    experiment = config["overlap_experiment"]


    strategy = cast(
        ChunkingStrategy,
        experiment["strategy"],
    )

    if strategy == "markdown":
        raise ValueError(
            "Pure Markdown chunking cannot be used "
            "for the overlap experiment because it "
            "has no controlled overlap."
        )

    chunk_size = int(
        experiment["chunk_size"]
    )

    configurations = [
        ChunkingConfiguration(
            name=(
                f"{strategy}"
                f"-size-{chunk_size}"
                f"-overlap-{int(chunk_overlap)}"
            ),
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_overlap),
        )
        for chunk_overlap
        in experiment["chunk_overlaps"]
    ]

    results: list[ConfigurationResult] = []

    for configuration in configurations:
        print(
            f"Running chunk overlap: "
            f"{configuration.chunk_overlap}"
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
                "chunk-overlap-experiment"
            ),
        )

        results.append(result)

    save_experiment_results(
        results=results,
        output_dir=output_dir,
    )

    return results