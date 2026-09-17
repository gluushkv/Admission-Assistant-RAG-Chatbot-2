from __future__ import annotations

from time import perf_counter

import torch

from rag.generation.generator import Generator


def measure_generation(
    *,
    generator: Generator,
    question: str,
    context: list[str],
) -> tuple[str, dict]:
    _synchronize_cuda()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start = perf_counter()

    raw_output = generator.generate(
        question=question,
        context=context,
    )

    _synchronize_cuda()

    elapsed_seconds = (
        perf_counter() - start
    )

    output_tokens = (
        generator.count_tokens(
            raw_output
        )
    )

    tokens_per_second = (
        output_tokens / elapsed_seconds
        if elapsed_seconds > 0
        else 0.0
    )

    peak_vram_bytes = 0

    if torch.cuda.is_available():
        peak_vram_bytes = int(
            torch.cuda.max_memory_allocated()
        )

    return raw_output, {
        "generation_latency_ms": (
            elapsed_seconds * 1000.0
        ),
        "output_tokens": (
            output_tokens
        ),
        "output_tokens_per_second": (
            tokens_per_second
        ),
        "peak_vram_bytes": (
            peak_vram_bytes
        ),
    }


def _synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()