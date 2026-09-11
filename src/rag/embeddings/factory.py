from __future__ import annotations

from functools import partial
from typing import Literal

from sentence_transformers import SentenceTransformer

from rag.embeddings.embedder import (
    SentenceTransformerEmbedder,
    TextFormatter,
)


EmbeddingModelName = Literal[
    "multilingual-e5-small",
    "multilingual-e5-base",
    "bge-m3",
    "qwen3-embedding-0.6b",
]


MODEL_IDS: dict[EmbeddingModelName, str] = {
    "multilingual-e5-small": (
        "intfloat/multilingual-e5-small"
    ),
    "multilingual-e5-base": (
        "intfloat/multilingual-e5-base"
    ),
    "bge-m3": (
        "BAAI/bge-m3"
    ),
    "qwen3-embedding-0.6b": (
        "Qwen/Qwen3-Embedding-0.6B"
    ),
}


QWEN_RETRIEVAL_INSTRUCTION = (
    "Given a question from a university applicant, retrieve "
    "relevant passages from official university admission "
    "documents that contain the information needed to answer "
    "the question."
)


def create_embedder(
    model_name: EmbeddingModelName,
    *,
    device: str | None = None,
    batch_size: int = 32,
    show_progress_bar: bool = False,
    qwen_instruction: str = QWEN_RETRIEVAL_INSTRUCTION,
) -> SentenceTransformerEmbedder:

    if model_name not in MODEL_IDS:
        raise ValueError(
            f"Unsupported embedding model: {model_name}"
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than 0"
        )

    model_id = MODEL_IDS[model_name]

    query_formatter: TextFormatter
    document_formatter: TextFormatter

    if model_name in {
        "multilingual-e5-small",
        "multilingual-e5-base",
    }:
        query_formatter = _format_e5_query
        document_formatter = _format_e5_document

    elif model_name == "bge-m3":
        query_formatter = _identity
        document_formatter = _identity

    elif model_name == "qwen3-embedding-0.6b":
        if not qwen_instruction.strip():
            raise ValueError(
                "Qwen retrieval instruction must not be empty"
            )

        query_formatter = partial(
            _format_qwen_query,
            instruction=qwen_instruction,
        )
        document_formatter = _identity

    else:
        raise ValueError(
            f"Unsupported embedding model: {model_name}"
        )

    model = SentenceTransformer(
        model_id,
        device=device,
    )

    return SentenceTransformerEmbedder(
        model=model,
        model_id=model_id,
        batch_size=batch_size,
        query_formatter=query_formatter,
        document_formatter=document_formatter,
        normalize_embeddings=True,
        show_progress_bar=show_progress_bar,
    )


def _format_e5_query(text: str) -> str:
    return f"query: {text}"


def _format_e5_document(text: str) -> str:
    return f"passage: {text}"


def _format_qwen_query(
    text: str,
    *,
    instruction: str,
) -> str:
    return (
        f"Instruct: {instruction}\n"
        f"Query:{text}"
    )


def _identity(text: str) -> str:
    return text