from __future__ import annotations

from collections.abc import Mapping, Sequence

from rag.embeddings.embedder import SentenceTransformerEmbedder
from rag.evaluation.models import GoldenQuestion
from rag.models import Chunk
from rag.retrieval.qdrant_store import QdrantStore


def retrieve_candidates(
    *,
    questions: Sequence[GoldenQuestion],
    embedder: SentenceTransformerEmbedder,
    store: QdrantStore,
    collection_name: str,
    candidate_k: int,
) -> dict[str, list[Chunk]]:

    if candidate_k <= 0:
        raise ValueError(
            "candidate_k must be greater than 0"
        )

    candidates: dict[str, list[Chunk]] = {}

    for question in questions:
        if not question.is_answerable:
            continue

        query_vectors = embedder.embed_queries(
            [question.question]
        )

        if len(query_vectors) != 1:
            raise RuntimeError(
                "Expected exactly one query embedding"
            )

        query_vector = query_vectors[0]

        candidates[
            question.question_id
        ] = store.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=candidate_k,
        )

    if len(candidates) == 0:
        raise RuntimeError(
            "Dense retrieval produced no candidate sets"
        )

    return candidates


def slice_candidates(
    candidates: Mapping[str, Sequence[Chunk]],
    *,
    candidate_k: int,
) -> dict[str, list[Chunk]]:

    if candidate_k <= 0:
        raise ValueError(
            "candidate_k must be greater than 0"
        )

    return {
        question_id: list(
            chunks[:candidate_k]
        )
        for question_id, chunks
        in candidates.items()
    }