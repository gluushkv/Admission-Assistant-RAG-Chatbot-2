from __future__ import annotations

from collections.abc import Sequence

from rag.chunking.factory import (
    ChunkingStrategy,
    create_chunker,
)
from rag.models import Chunk, Document


def build_chunks(
    documents: Sequence[Document],
    *,
    strategy: ChunkingStrategy,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    sentence_language: str = "russian",
) -> list[Chunk]:

    if not documents:
        raise ValueError(
            "documents must not be empty."
        )

    chunker = create_chunker(
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        sentence_language=sentence_language,
    )

    chunks: list[Chunk] = []

    for document in documents:
        document_chunks = chunker.chunk(document)

        if not document_chunks:
            raise RuntimeError(
                f"Chunker produced no chunks for document "
                f"'{document.document_id}'."
            )

        for chunk in document_chunks:
            _validate_chunk(
                document=document,
                chunk=chunk,
            )

        chunks.extend(document_chunks)

    _validate_unique_chunk_ids(chunks)

    return chunks


def _validate_chunk(
    *,
    document: Document,
    chunk: Chunk,
) -> None:

    if chunk.document_id != document.document_id:
        raise ValueError(
            f"Chunk '{chunk.chunk_id}' belongs to document "
            f"'{chunk.document_id}', expected "
            f"'{document.document_id}'."
        )

    if chunk.start_offset < 0:
        raise ValueError(
            f"Chunk '{chunk.chunk_id}' has negative start_offset."
        )

    if chunk.end_offset <= chunk.start_offset:
        raise ValueError(
            f"Chunk '{chunk.chunk_id}' has invalid offsets: "
            f"[{chunk.start_offset}, {chunk.end_offset})."
        )

    if chunk.end_offset > len(document.text):
        raise ValueError(
            f"Chunk '{chunk.chunk_id}' ends outside document "
            f"'{document.document_id}'."
        )

    source_text = document.text[
        chunk.start_offset:chunk.end_offset
    ]

    if source_text != chunk.text:
        raise ValueError(
            f"Chunk '{chunk.chunk_id}' text does not match "
            "document text at its offsets."
        )


def _validate_unique_chunk_ids(
    chunks: Sequence[Chunk],
) -> None:

    seen_ids: set[str] = set()

    for chunk in chunks:
        if chunk.chunk_id in seen_ids:
            raise ValueError(
                f"Duplicate chunk_id: {chunk.chunk_id}"
            )

        seen_ids.add(chunk.chunk_id)