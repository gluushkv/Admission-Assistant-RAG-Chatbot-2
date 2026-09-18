from __future__ import annotations

import gc
from collections.abc import Sequence

import torch

from rag.chunking.builder import build_chunks
from rag.embeddings.factory import create_embedder
from rag.evaluation.models import GoldenQuestion
from rag.models import Chunk, Document
from rag.reranking.factory import create_reranker
from rag.retrieval.indexing import build_dense_index
from rag.retrieval.qdrant_store import QdrantStore
from rag.retrieval.retriever import DenseRetriever


CANDIDATE_K = 10
FINAL_K = 5


def build_gold_contexts(
    questions: Sequence[GoldenQuestion],
) -> dict[str, list[Chunk]]:
    contexts: dict[str, list[Chunk]] = {}

    for question in questions:
        question_chunks: list[Chunk] = []

        for evidence in question.evidence:
            if evidence.text is None:
                raise ValueError(
                    f"Evidence '{evidence.evidence_id}' "
                    "does not contain text."
                )

            question_chunks.append(
                Chunk(
                    chunk_id=(
                        f"gold:{evidence.evidence_id}"
                    ),
                    document_id=evidence.document_id,
                    text=evidence.text,
                    start_offset=evidence.start_offset,
                    end_offset=evidence.end_offset,
                )
            )

        contexts[
            question.question_id
        ] = question_chunks

    return contexts


def build_retrieved_contexts(
    *,
    documents: Sequence[Document],
    questions: Sequence[GoldenQuestion],
    store: QdrantStore,
    device: str = "cuda",
    candidate_k: int = CANDIDATE_K,
    final_k: int = FINAL_K,
) -> dict[str, list[Chunk]]:
    if not documents:
        raise ValueError(
            "documents must not be empty"
        )

    if not questions:
        raise ValueError(
            "questions must not be empty"
        )

    if candidate_k <= 0:
        raise ValueError(
            "candidate_k must be greater than 0"
        )

    if final_k <= 0:
        raise ValueError(
            "final_k must be greater than 0"
        )

    if final_k > candidate_k:
        raise ValueError(
            "final_k must not exceed candidate_k"
        )


    chunks = build_chunks(
        documents=documents,
        strategy="markdown",
    )

    embedder = None
    retriever = None
    reranker = None

    try:

        embedder = create_embedder(
            model_name="bge-m3",
            device=device,
            batch_size=32,
        )

        collection_name = (
            "generation-retrieved-context"
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


        reranker = create_reranker(
            model_id="BAAI/bge-reranker-v2-m3",
            device=device,
            batch_size=16,
        )

        contexts: dict[
            str,
            list[Chunk],
        ] = {}

        for index, question in enumerate(
            questions,
            start=1,
        ):
            candidates = retriever.retrieve(
                question.question,
                k=candidate_k,
            )

            reranked = reranker.rerank(
                question.question,
                candidates,
                top_k=final_k,
            )

            contexts[
                question.question_id
            ] = reranked

            print(
                f"[{index}/{len(questions)}] "
                f"{question.question_id}: "
                f"{len(reranked)} chunks"
            )

        return contexts

    finally:
        reranker = None
        retriever = None
        embedder = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()