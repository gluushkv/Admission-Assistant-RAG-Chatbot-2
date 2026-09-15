from __future__ import annotations

from collections.abc import Sequence
from statistics import fmean, median

import numpy as np

from experiments.chunking.models import ChunkStatistics
from rag.models import Chunk, Document


DEFAULT_PERCENTILES = (90, 95, 99)
DEFAULT_THRESHOLDS = (1000, 1500, 2000)


def compute_chunk_statistics(
    *,
    documents: Sequence[Document],
    chunks: Sequence[Chunk],
    percentiles: tuple[int, ...] = DEFAULT_PERCENTILES,
    thresholds: tuple[int, ...] = DEFAULT_THRESHOLDS,
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

    n_chunks = len(chunks)
    total_indexed_characters = sum(lengths)

    percentile_values = {
        percentile: float(
            np.percentile(lengths, percentile)
        )
        for percentile in percentiles
    }

    counts_above = {
        threshold: sum(
            length > threshold
            for length in lengths
        )
        for threshold in thresholds
    }

    shares_above = {
        threshold: count / n_chunks
        for threshold, count in counts_above.items()
    }

    return ChunkStatistics(
        n_chunks=n_chunks,
        total_indexed_characters=total_indexed_characters,
        mean_length=fmean(lengths),
        median_length=median(lengths),
        min_length=min(lengths),
        max_length=max(lengths),
        percentiles=percentile_values,
        counts_above=counts_above,
        shares_above=shares_above,
        indexed_text_ratio=(
            total_indexed_characters
            / total_document_characters
        ),
    )