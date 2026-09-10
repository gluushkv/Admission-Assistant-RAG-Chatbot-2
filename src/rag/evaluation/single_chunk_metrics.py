from collections.abc import Sequence

from rag.models import Chunk
from rag.evaluation.evidence_matching import single_chunk_match
from rag.evaluation.models import Evidence, GoldenQuestion


def _evidence_found(
    evidence: Evidence,
    chunks: Sequence[Chunk],
) -> bool:
    return any(
        single_chunk_match(evidence, chunk)
        for chunk in chunks
    )


def single_chunk_hit_at_k(
    question: GoldenQuestion,
    chunks: Sequence[Chunk],
    k: int,
) -> float:

    top_k = chunks[:k]

    return float(
        any(
            _evidence_found(evidence, top_k)
            for evidence in question.evidence
        )
    )


def single_chunk_recall_at_k(
    question: GoldenQuestion,
    chunks: Sequence[Chunk],
    k: int,
) -> float:

    top_k = chunks[:k]

    found = [
        _evidence_found(evidence, top_k)
        for evidence in question.evidence
    ]

    if question.evidence_relation is None:
        return float(found[0])

    if question.evidence_relation == "OR":
        return float(any(found))

    if question.evidence_relation == "AND":
        return sum(found) / len(found)

    raise ValueError(
        f"Unsupported evidence relation: "
        f"{question.evidence_relation}"
    )


def single_chunk_reciprocal_rank_at_k(
    question: GoldenQuestion,
    chunks: Sequence[Chunk],
    k: int,
) -> float:

    for rank, chunk in enumerate(chunks[:k], start=1):
        if any(
            single_chunk_match(evidence, chunk)
            for evidence in question.evidence
        ):
            return 1.0 / rank

    return 0.0