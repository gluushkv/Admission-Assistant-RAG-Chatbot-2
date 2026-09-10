from collections.abc import Sequence

from rag.models import Chunk
from rag.evaluation.models import Evidence


def single_chunk_match(
    evidence: Evidence,
    chunk: Chunk,
) -> bool:

    if evidence.document_id != chunk.document_id:
        return False

    return (
        chunk.start_offset <= evidence.start_offset
        and chunk.end_offset >= evidence.end_offset
    )


def _merge_intervals(
    intervals: Sequence[tuple[int, int]],
) -> list[tuple[int, int]]:

    if not intervals:
        return []

    sorted_intervals = sorted(intervals)

    merged: list[tuple[int, int]] = []
    current_start, current_end = sorted_intervals[0]

    for start, end in sorted_intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end

    merged.append((current_start, current_end))

    return merged


def evidence_context_coverage(
    evidence: Evidence,
    chunks: Sequence[Chunk],
) -> float:

    covered_intervals: list[tuple[int, int]] = []

    for chunk in chunks:
        if chunk.document_id != evidence.document_id:
            continue

        overlap_start = max(
            evidence.start_offset,
            chunk.start_offset,
        )
        overlap_end = min(
            evidence.end_offset,
            chunk.end_offset,
        )

        if overlap_start < overlap_end:
            covered_intervals.append(
                (overlap_start, overlap_end)
            )

    merged_intervals = _merge_intervals(covered_intervals)

    covered_length = sum(
        end - start
        for start, end in merged_intervals
    )

    evidence_length = (
        evidence.end_offset - evidence.start_offset
    )

    return covered_length / evidence_length


def context_match(
    evidence: Evidence,
    chunks: Sequence[Chunk],
) -> bool:

    return evidence_context_coverage(evidence, chunks) == 1.0