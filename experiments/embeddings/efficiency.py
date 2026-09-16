from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter

import numpy as np
import torch

from rag.embeddings.embedder import (
    SentenceTransformerEmbedder,
)


def measure_embedding_efficiency(
    *,
    embedder: SentenceTransformerEmbedder,
    corpus_texts: Sequence[str],
    queries: Sequence[str],
    device: str = "cuda",
    warmup_documents: int = 32,
    warmup_queries: int = 5,
    query_repeats: int = 1,
) -> tuple[np.ndarray, dict]:

    if len(corpus_texts) == 0:
        raise ValueError(
            "corpus_texts must not be empty"
        )

    if len(queries) == 0:
        raise ValueError(
            "queries must not be empty"
        )

    if warmup_documents < 0:
        raise ValueError(
            "warmup_documents must be non-negative"
        )

    if warmup_queries < 0:
        raise ValueError(
            "warmup_queries must be non-negative"
        )

    if query_repeats <= 0:
        raise ValueError(
            "query_repeats must be greater than 0"
        )

    cuda_device = _validate_cuda_device(
        device
    )


    if warmup_documents > 0:
        n_warmup_documents = min(
            warmup_documents,
            len(corpus_texts),
        )

        embedder.embed_documents(
            corpus_texts[
                :n_warmup_documents
            ]
        )

        _synchronize_cuda(
            cuda_device
        )

    _reset_peak_vram(
        cuda_device
    )

    _synchronize_cuda(
        cuda_device
    )

    corpus_start = perf_counter()

    vectors = embedder.embed_documents(
        corpus_texts
    )

    _synchronize_cuda(
        cuda_device
    )

    corpus_embedding_time_seconds = (
        perf_counter()
        - corpus_start
    )

    corpus_peak_vram_bytes = (
        _peak_vram_bytes(
            cuda_device
        )
    )

    vectors_array = np.asarray(
        vectors
    )

    if vectors_array.ndim != 2:
        raise RuntimeError(
            "Expected corpus embeddings "
            "to be a 2D array"
        )

    if len(vectors_array) != len(
        corpus_texts
    ):
        raise RuntimeError(
            "Number of corpus embeddings does not "
            "match number of corpus texts"
        )

    if corpus_embedding_time_seconds <= 0:
        raise RuntimeError(
            "Corpus embedding time must be positive"
        )

    corpus_embedding_throughput = (
        len(corpus_texts)
        / corpus_embedding_time_seconds
    )

    if warmup_queries > 0:
        n_warmup_queries = min(
            warmup_queries,
            len(queries),
        )

        for query in queries[
            :n_warmup_queries
        ]:
            query_vectors = (
                embedder.embed_queries(
                    [query]
                )
            )

            if len(query_vectors) != 1:
                raise RuntimeError(
                    "Expected exactly one query embedding"
                )

        _synchronize_cuda(
            cuda_device
        )

    _reset_peak_vram(
        cuda_device
    )

    query_latencies_ms: list[
        float
    ] = []

    for _ in range(
        query_repeats
    ):
        for query in queries:

            _synchronize_cuda(
                cuda_device
            )

            start = perf_counter()

            query_vectors = (
                embedder.embed_queries(
                    [query]
                )
            )

            _synchronize_cuda(
                cuda_device
            )

            elapsed_ms = (
                perf_counter()
                - start
            ) * 1000.0

            if len(query_vectors) != 1:
                raise RuntimeError(
                    "Expected exactly one query embedding"
                )

            query_latencies_ms.append(
                elapsed_ms
            )

    query_peak_vram_bytes = (
        _peak_vram_bytes(
            cuda_device
        )
    )

    if len(query_latencies_ms) == 0:
        raise RuntimeError(
            "No query latency measurements "
            "were produced"
        )

    metrics = {
        "embedding_dimension": (
            int(
                vectors_array.shape[1]
            )
        ),
        "corpus_embedding_time_seconds": (
            float(
                corpus_embedding_time_seconds
            )
        ),
        "corpus_embedding_throughput_chunks_per_sec": (
            float(
                corpus_embedding_throughput
            )
        ),
        "median_query_embedding_latency_ms": (
            float(
                np.median(
                    query_latencies_ms
                )
            )
        ),
        "p95_query_embedding_latency_ms": (
            float(
                np.percentile(
                    query_latencies_ms,
                    95,
                )
            )
        ),
        "corpus_peak_vram_bytes": (
            corpus_peak_vram_bytes
        ),
        "query_peak_vram_bytes": (
            query_peak_vram_bytes
        ),
        "raw_vector_size_bytes": (
            int(
                vectors_array.nbytes
            )
        ),
        "n_corpus_chunks": (
            len(corpus_texts)
        ),
        "n_queries": (
            len(queries)
        ),
        "n_query_latency_measurements": (
            len(query_latencies_ms)
        ),
        "warmup_documents": (
            warmup_documents
        ),
        "warmup_queries": (
            warmup_queries
        ),
        "query_repeats": (
            query_repeats
        ),
    }

    return (
        vectors_array,
        metrics,
    )


def _validate_cuda_device(
    device: str,
) -> torch.device:

    cuda_device = torch.device(
        device
    )

    if cuda_device.type != "cuda":
        raise ValueError(
            "Embedding efficiency experiment "
            "must run on CUDA"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available"
        )

    return cuda_device


def _synchronize_cuda(
    device: torch.device,
) -> None:

    torch.cuda.synchronize(
        device
    )


def _reset_peak_vram(
    device: torch.device,
) -> None:

    torch.cuda.reset_peak_memory_stats(
        device
    )


def _peak_vram_bytes(
    device: torch.device,
) -> int:

    return int(
        torch.cuda.max_memory_allocated(
            device
        )
    )