from __future__ import annotations

from collections.abc import Sequence
from statistics import fmean, median

from experiments.chunking.models import ChunkStatistics
from rag.models import Chunk, Document


def compute_chunk_statistics(
    *,
    documents: Sequence[Document],
    chunks: Sequence[Chunk],
) -> ChunkStatistics:

    if not chunks:
        raise ValueError(
            "Cannot calculate statistics for an empty chunk collection."
        )

    lengths = [
        len(chunk.text)
        for chunk in chunks
    ]

    total_document_characters = sum(
        len(document.text)
        for document in documents
    )

    if total_document_characters <= 0:
        raise ValueError(
            "Total document length must be greater than zero."
        )

    total_indexed_characters = sum(lengths)

    return ChunkStatistics(
        n_chunks=len(chunks),
        total_indexed_characters=total_indexed_characters,
        mean_length=fmean(lengths),
        median_length=median(lengths),
        min_length=min(lengths),
        max_length=max(lengths),
        indexed_text_ratio=(
            total_indexed_characters
            / total_document_characters
        ),
    )