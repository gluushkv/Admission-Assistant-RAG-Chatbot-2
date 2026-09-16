from __future__ import annotations

from collections.abc import Mapping, Sequence
from time import perf_counter

import numpy as np
import torch

from rag.evaluation.models import GoldenQuestion
from rag.models import Chunk
from rag.reranking.reranker import (
    CrossEncoderReranker,
)


def measure_reranking_efficiency(
    *,
    reranker: CrossEncoderReranker,
    questions: Sequence[GoldenQuestion],
    candidates: Mapping[str, Sequence[Chunk]],
    final_k: int,
    warmup_queries: int = 3,
) -> tuple[dict[str, list[Chunk]], dict]:

    if final_k <= 0:
        raise ValueError(
            "final_k must be greater than 0"
        )

    if warmup_queries < 0:
        raise ValueError(
            "warmup_queries must be non-negative"
        )

    answerable_questions = [
        question
        for question in questions
        if question.is_answerable
    ]

    if len(answerable_questions) == 0:
        raise ValueError(
            "No answerable questions were provided"
        )

    for question in answerable_questions:
        if question.question_id not in candidates:
            raise ValueError(
                "Missing candidates for question "
                f"'{question.question_id}'"
            )

    for question in answerable_questions[
        :warmup_queries
    ]:
        reranker.rerank(
            question.question,
            candidates[
                question.question_id
            ],
            top_k=final_k,
        )

    _synchronize_cuda()
    _reset_peak_vram()

    rankings: dict[str, list[Chunk]] = {}
    latencies_ms: list[float] = []


    for question in answerable_questions:
        question_candidates = candidates[
            question.question_id
        ]

        _synchronize_cuda()

        start = perf_counter()

        ranked_chunks = reranker.rerank(
            question.question,
            question_candidates,
            top_k=final_k,
        )

        _synchronize_cuda()

        elapsed_ms = (
            perf_counter()
            - start
        ) * 1000.0

        rankings[
            question.question_id
        ] = ranked_chunks

        latencies_ms.append(
            elapsed_ms
        )

    peak_vram_bytes = (
        _peak_vram_bytes()
    )

    metrics = {
        "model_id": reranker.model_id,
        "n_queries": len(latencies_ms),
        "median_reranking_latency_ms": float(
            np.median(
                latencies_ms
            )
        ),
        "p95_reranking_latency_ms": float(
            np.percentile(
                latencies_ms,
                95,
            )
        ),
        "peak_vram_bytes": (
            peak_vram_bytes
        ),
    }

    return rankings, metrics


def _synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset_peak_vram() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _peak_vram_bytes() -> int:
    if not torch.cuda.is_available():
        return 0

    return int(
        torch.cuda.max_memory_allocated()
    )