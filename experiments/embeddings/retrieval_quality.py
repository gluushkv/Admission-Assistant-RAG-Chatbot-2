from __future__ import annotations

from collections.abc import Sequence
from statistics import fmean

import numpy as np

from rag.embeddings.embedder import SentenceTransformerEmbedder
from rag.evaluation.evaluator import (
    evaluate_retrieval_question,
)
from rag.evaluation.models import GoldenQuestion
from rag.models import Chunk
from rag.retrieval.qdrant_store import QdrantStore


FORMULATION_TYPES = {
    "clean",
    "colloquial",
    "abbreviation",
    "typo",
}


def run_retrieval_quality(
    *,
    chunks: Sequence[Chunk],
    vectors: np.ndarray,
    questions: Sequence[GoldenQuestion],
    embedder: SentenceTransformerEmbedder,
    store: QdrantStore,
    configuration_name: str,
    ks: tuple[int, ...] = (1, 3, 5),
    collection_name: str = "embedding-experiment",
) -> list[dict]:

    if len(chunks) == 0:
        raise ValueError(
            "chunks must not be empty"
        )

    if vectors.ndim != 2:
        raise ValueError(
            "vectors must be a 2D array"
        )

    if len(vectors) != len(chunks):
        raise ValueError(
            "Number of vectors must match "
            "number of chunks"
        )

    if len(questions) == 0:
        raise ValueError(
            "questions must not be empty"
        )

    if len(ks) == 0:
        raise ValueError(
            "ks must not be empty"
        )

    if any(k <= 0 for k in ks):
        raise ValueError(
            "All k values must be greater than 0"
        )

    store.create_collection(
        collection_name=collection_name,
        vector_size=int(
            vectors.shape[1]
        ),
        recreate=True,
    )

    store.upsert_chunks(
        collection_name=collection_name,
        chunks=chunks,
        vectors=vectors,
    )

    retrieval_k = max(ks)

    records: list[dict] = []

    for question in questions:

        if not question.is_answerable:
            continue

        query_vectors = (
            embedder.embed_queries(
                [question.question]
            )
        )

        if len(query_vectors) != 1:
            raise RuntimeError(
                "Expected exactly one query embedding"
            )

        query_vector = query_vectors[0]

        retrieved_chunks = store.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=retrieval_k,
        )

        evaluation = (
            evaluate_retrieval_question(
                question=question,
                chunks=retrieved_chunks,
                ks=ks,
            )
        )

        records.append(
            {
                "configuration": (
                    configuration_name
                ),
                "model_id": (
                    embedder.model_id
                ),
                "question_id": (
                    question.question_id
                ),
                "formulation_type": (
                    question.formulation_type
                ),
                "evidence_relation": (
                    question.evidence_relation
                ),
                "retrieved_chunk_ids": [
                    chunk.chunk_id
                    for chunk in retrieved_chunks
                ],
                "single_chunk_metrics": (
                    evaluation[
                        "single_chunk_metrics"
                    ]
                ),
                "context_metrics": (
                    evaluation[
                        "context_metrics"
                    ]
                ),
            }
        )

    if len(records) == 0:
        raise RuntimeError(
            "Retrieval experiment produced "
            "no answerable-question results"
        )

    return records


def build_retrieval_quality_summary(
    records: Sequence[dict],
) -> list[dict]:

    if len(records) == 0:
        raise ValueError(
            "records must not be empty"
        )

    scopes: dict[
        str,
        list[dict],
    ] = {
        "all_answerable": list(
            records
        ),
        "and_evidence": [
            record
            for record in records
            if record[
                "evidence_relation"
            ] == "AND"
        ],
    }

    for formulation_type in sorted(
        FORMULATION_TYPES
    ):
        scopes[
            f"formulation_{formulation_type}"
        ] = [
            record
            for record in records
            if record[
                "formulation_type"
            ] == formulation_type
        ]

    rows: list[dict] = []

    for (
        scope_name,
        scope_records,
    ) in scopes.items():

        if len(scope_records) == 0:
            continue

        first_record = scope_records[0]

        row = {
            "configuration": (
                first_record[
                    "configuration"
                ]
            ),
            "model_id": (
                first_record[
                    "model_id"
                ]
            ),
            "question_scope": (
                scope_name
            ),
            "n_questions": (
                len(scope_records)
            ),
        }

        row.update(
            _aggregate_metrics(
                scope_records
            )
        )

        rows.append(
            row
        )

    return rows


def _aggregate_metrics(
    records: Sequence[dict],
) -> dict[str, float]:

    if len(records) == 0:
        raise ValueError(
            "Cannot aggregate empty records"
        )

    result: dict[
        str,
        float,
    ] = {}

    first_record = records[0]

    single_chunk_metric_names = (
        first_record[
            "single_chunk_metrics"
        ].keys()
    )

    context_metric_names = (
        first_record[
            "context_metrics"
        ].keys()
    )

    for metric_name in (
        single_chunk_metric_names
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
        context_metric_names
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