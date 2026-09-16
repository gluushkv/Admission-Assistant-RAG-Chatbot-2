from __future__ import annotations

from collections.abc import Mapping, Sequence
from statistics import fmean

from rag.evaluation.context_metrics import (
    context_recall_at_k,
)
from rag.evaluation.evaluator import (
    evaluate_retrieval_question,
)
from rag.evaluation.models import GoldenQuestion
from rag.models import Chunk


def build_baseline_rankings(
    candidates: Mapping[str, Sequence[Chunk]],
    *,
    final_k: int,
) -> dict[str, list[Chunk]]:

    if final_k <= 0:
        raise ValueError(
            "final_k must be greater than 0"
        )

    return {
        question_id: list(
            chunks[:final_k]
        )
        for question_id, chunks
        in candidates.items()
    }


def evaluate_rankings(
    *,
    questions: Sequence[GoldenQuestion],
    rankings: Mapping[str, Sequence[Chunk]],
    configuration_name: str,
    ks: tuple[int, ...] = (1, 3, 5),
) -> list[dict]:

    if len(ks) == 0:
        raise ValueError(
            "ks must not be empty"
        )

    if any(k <= 0 for k in ks):
        raise ValueError(
            "All k values must be greater than 0"
        )

    records: list[dict] = []

    for question in questions:
        if not question.is_answerable:
            continue

        if question.question_id not in rankings:
            raise ValueError(
                "Missing ranking for question "
                f"'{question.question_id}'"
            )

        ranked_chunks = rankings[
            question.question_id
        ]

        evaluation = evaluate_retrieval_question(
            question=question,
            chunks=ranked_chunks,
            ks=ks,
        )

        records.append(
            {
                "configuration": (
                    configuration_name
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
                    for chunk in ranked_chunks
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
            "Evaluation produced no records"
        )

    return records


def compute_candidate_recall(
    *,
    questions: Sequence[GoldenQuestion],
    candidates: Mapping[str, Sequence[Chunk]],
    candidate_k: int,
) -> dict[str, float]:

    if candidate_k <= 0:
        raise ValueError(
            "candidate_k must be greater than 0"
        )

    result: dict[str, float] = {}

    for question in questions:
        if not question.is_answerable:
            continue

        if question.question_id not in candidates:
            raise ValueError(
                "Missing candidates for question "
                f"'{question.question_id}'"
            )

        chunks = candidates[
            question.question_id
        ]

        result[
            question.question_id
        ] = float(
            context_recall_at_k(
                question,
                chunks,
                candidate_k,
            )
        )

    return result


def aggregate_retrieval_quality(
    records: Sequence[dict],
) -> dict[str, float]:

    if len(records) == 0:
        raise ValueError(
            "records must not be empty"
        )

    result: dict[str, float] = {}

    single_chunk_metrics = (
        records[0][
            "single_chunk_metrics"
        ].keys()
    )

    context_metrics = (
        records[0][
            "context_metrics"
        ].keys()
    )

    for metric_name in single_chunk_metrics:
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

    for metric_name in context_metrics:
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