from __future__ import annotations

from rag.reranking.reranker import (
    CrossEncoderReranker,
)


def create_reranker(
    model_id: str,
    *,
    device: str = "cuda",
    batch_size: int = 16,
    trust_remote_code: bool = False,
) -> CrossEncoderReranker:
    return CrossEncoderReranker(
        model_id=model_id,
        device=device,
        batch_size=batch_size,
        trust_remote_code=trust_remote_code,
    )