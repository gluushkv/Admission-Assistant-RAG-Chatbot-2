from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Document:
    document_id: str
    title: str
    text: str
    source_url: str | None
    source_path: Path

@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    start_offset: int
    end_offset: int
