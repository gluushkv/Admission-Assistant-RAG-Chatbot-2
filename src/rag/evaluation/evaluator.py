from collections.abc import Sequence

from rag.models import Chunk
from rag.evaluation.context_metrics import context_recall_at_k
from rag.evaluation.models import GoldenQuestion
from rag.evaluation.single_chunk_metrics import (
    single_chunk_hit_at_k,
    single_chunk_recall_at_k,
    single_chunk_reciprocal_rank_at_k,
)


DEFAULT_KS = (1, 3, 5)


def evaluate_retrieval_question(
    question: GoldenQuestion,
    chunks: Sequence[Chunk],
    ks: tuple[int, ...] = DEFAULT_KS,
) -> dict:

    if not question.is_answerable:
        raise ValueError(
            "Retrieval metrics are defined only "
            "for answerable questions."
        )

    result = {
        "question_id": question.question_id,
        "formulation_type": question.formulation_type,
        "single_chunk_metrics": {},
        "context_metrics": {},
    }

    for k in ks:
        result["single_chunk_metrics"][f"hit@{k}"] = (
            single_chunk_hit_at_k(question, chunks, k)
        )

        result["single_chunk_metrics"][f"recall@{k}"] = (
            single_chunk_recall_at_k(question, chunks, k)
        )

        result["single_chunk_metrics"][f"mrr@{k}"] = (
            single_chunk_reciprocal_rank_at_k(
                question,
                chunks,
                k,
            )
        )

        result["context_metrics"][f"recall@{k}"] = (
            context_recall_at_k(question, chunks, k)
        )

    return result