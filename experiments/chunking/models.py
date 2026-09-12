from __future__ import annotations

from dataclasses import dataclass

from rag.chunking.factory import ChunkingStrategy


@dataclass(frozen=True)
class ChunkingConfiguration:

    name: str
    strategy: ChunkingStrategy
    chunk_size: int
    chunk_overlap: int


@dataclass(frozen=True)
class ChunkStatistics:

    n_chunks: int
    total_indexed_characters: int

    mean_length: float
    median_length: float
    std_length: float

    min_length: int
    max_length: int

    indexed_text_ratio: float


@dataclass(frozen=True)
class ConfigurationResult:

    configuration: ChunkingConfiguration
    embedding_model: str
    chunk_statistics: ChunkStatistics
    records: list[dict]