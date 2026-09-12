from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from statistics import fmean

import yaml

from experiments.chunking.models import (
    ChunkingConfiguration,
    ConfigurationResult,
)
from experiments.chunking.statistics import (
    compute_chunk_statistics,
)
from rag.chunking.builder import build_chunks
from rag.embeddings.embedder import SentenceTransformerEmbedder
from rag.evaluation.evaluator import evaluate_retrieval_question
from rag.evaluation.models import GoldenQuestion
from rag.models import Document
from rag.retrieval.indexing import build_dense_index
from rag.retrieval.qdrant_store import QdrantStore
from rag.retrieval.retriever import DenseRetriever


FORMULATION_TYPES = {
    "clean",
    "colloquial",
    "abbreviation",
    "typo",
}


def load_chunking_config(
    path: str | Path,
) -> dict:

    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            f"Expected a mapping in config file: {path}"
        )

    return config


def run_chunking_configuration(
    *,
    documents: Sequence[Document],
    questions: Sequence[GoldenQuestion],
    configuration: ChunkingConfiguration,
    embedder: SentenceTransformerEmbedder,
    store: QdrantStore,
    ks: tuple[int, ...] = (1, 3, 5),
    sentence_language: str = "russian",
    collection_name: str = "chunking-experiment",
) -> ConfigurationResult:

    if not ks:
        raise ValueError(
            "ks must not be empty."
        )

    if any(k <= 0 for k in ks):
        raise ValueError(
            "All k values must be greater than zero."
        )

    chunks = build_chunks(
        documents=list(documents),
        strategy=configuration.strategy,
        chunk_size=configuration.chunk_size,
        chunk_overlap=configuration.chunk_overlap,
        sentence_language=sentence_language,
    )

    chunk_statistics = compute_chunk_statistics(
        documents=documents,
        chunks=chunks,
    )

    build_dense_index(
        chunks=chunks,
        embedder=embedder,
        store=store,
        collection_name=collection_name,
        recreate=True,
    )

    retriever = DenseRetriever(
        embedder=embedder,
        store=store,
        collection_name=collection_name,
    )

    retrieval_k = max(ks)

    records: list[dict] = []

    for question in questions:

        if not question.is_answerable:
            continue

        retrieved_chunks = retriever.retrieve(
            question.question,
            k=retrieval_k,
        )

        evaluation = evaluate_retrieval_question(
            question=question,
            chunks=retrieved_chunks,
            ks=ks,
        )

        records.append(
            {
                "configuration": configuration.name,
                "strategy": configuration.strategy,
                "chunk_size": _reported_chunk_size(
                    configuration
                ),
                "chunk_overlap": _reported_chunk_overlap(
                    configuration
                ),
                "embedding_model": embedder.model_id,
                "question_id": question.question_id,
                "formulation_type": question.formulation_type,
                "evidence_relation": question.evidence_relation,
                "retrieved_chunk_ids": [
                    chunk.chunk_id
                    for chunk in retrieved_chunks
                ],
                "single_chunk_metrics": (
                    evaluation["single_chunk_metrics"]
                ),
                "context_metrics": (
                    evaluation["context_metrics"]
                ),
            }
        )

    if not records:
        raise RuntimeError(
            "Experiment produced no results for "
            "answerable questions."
        )

    return ConfigurationResult(
        configuration=configuration,
        embedding_model=embedder.model_id,
        chunk_statistics=chunk_statistics,
        records=records,
    )


def save_experiment_results(
    *,
    results: Sequence[ConfigurationResult],
    output_dir: str | Path,
) -> None:

    if not results:
        raise ValueError(
            "Cannot save empty experiment results."
        )

    output_dir = Path(output_dir)

    raw_dir = output_dir / "raw"
    tables_dir = output_dir / "tables"

    raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    tables_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    _save_raw_results(
        results=results,
        output_path=raw_dir / "results.jsonl",
    )

    summary = build_summary(results)

    _save_summary(
        rows=summary,
        output_path=tables_dir / "summary.csv",
    )


def build_summary(
    results: Sequence[ConfigurationResult],
) -> list[dict]:

    rows: list[dict] = []

    for result in results:
        records = result.records

        scopes = {
            "all_answerable": list(records),

            "and_evidence": [
                record
                for record in records
                if record["evidence_relation"] == "AND"
            ],
        }

        for formulation_type in sorted(FORMULATION_TYPES):
            scopes[
                f"formulation_{formulation_type}"
            ] = [
                record
                for record in records
                if record["formulation_type"]
                == formulation_type
            ]

        for scope_name, scope_records in scopes.items():
            if not scope_records:
                continue

            row = {
                "configuration": (
                    result.configuration.name
                ),
                "strategy": (
                    result.configuration.strategy
                ),
                "chunk_size": _reported_chunk_size(
                    result.configuration
                ),
                "chunk_overlap": _reported_chunk_overlap(
                    result.configuration
                ),
                "embedding_model": (
                    result.embedding_model
                ),
                "question_scope": scope_name,
                "n_questions": len(scope_records),
                "n_chunks": (
                    result.chunk_statistics.n_chunks
                ),
                "total_indexed_characters": (
                    result.chunk_statistics
                    .total_indexed_characters
                ),
                "mean_chunk_length": (
                    result.chunk_statistics.mean_length
                ),
                "median_chunk_length": (
                    result.chunk_statistics.median_length
                ),
                "std_chunk_length": (
                    result.chunk_statistics.std_length
                ),
                "min_chunk_length": (
                    result.chunk_statistics.min_length
                ),
                "max_chunk_length": (
                    result.chunk_statistics.max_length
                ),
                "indexed_text_ratio": (
                    result.chunk_statistics.indexed_text_ratio
                ),
            }

            row.update(
                _aggregate_metrics(
                    scope_records
                )
            )

            rows.append(row)

    return rows


def _aggregate_metrics(
    records: Sequence[dict],
) -> dict[str, float]:

    if not records:
        raise ValueError(
            "Cannot aggregate empty records."
        )

    result: dict[str, float] = {}

    first_record = records[0]

    for metric_name in (
        first_record["single_chunk_metrics"]
    ):
        result[
            f"single_chunk_{metric_name}"
        ] = fmean(
            float(
                record[
                    "single_chunk_metrics"
                ][metric_name]
            )
            for record in records
        )

    for metric_name in (
        first_record["context_metrics"]
    ):
        result[
            f"context_{metric_name}"
        ] = fmean(
            float(
                record[
                    "context_metrics"
                ][metric_name]
            )
            for record in records
        )

    return result


def _save_raw_results(
    *,
    results: Sequence[ConfigurationResult],
    output_path: Path,
) -> None:

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for result in results:
            for record in result.records:
                file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                )
                file.write("\n")


def _save_summary(
    *,
    rows: Sequence[dict],
    output_path: Path,
) -> None:

    if not rows:
        raise ValueError(
            "Experiment produced no summary rows."
        )

    fieldnames: list[str] = []

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def _reported_chunk_size(
    configuration: ChunkingConfiguration,
) -> int | None:
    if configuration.strategy == "markdown":
        return None

    return configuration.chunk_size


def _reported_chunk_overlap(
    configuration: ChunkingConfiguration,
) -> int | None:
    if configuration.strategy == "markdown":
        return None

    return configuration.chunk_overlap