from collections.abc import Sequence

from rag.models import Chunk
from rag.evaluation.evidence_matching import context_match
from rag.evaluation.models import GoldenQuestion


def context_recall_at_k(
    question: GoldenQuestion,
    chunks: Sequence[Chunk],
    k: int,
) -> float:

    top_k = chunks[:k]

    found = [
        context_match(evidence, top_k)
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