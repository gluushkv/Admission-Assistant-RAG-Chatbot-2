from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from rag.models import Chunk, Document


class Chunker(Protocol):
    """Common interface for all chunking strategies."""

    def chunk(self, document: Document) -> list[Chunk]:
        """Split one document into ordered chunks."""
        ...


class StringTextSplitter(Protocol):
    """Minimal interface required from text splitters returning strings."""

    def split_text(self, text: str) -> list[str]:
        ...


class SplitDocument(Protocol):
    """Minimal interface required from document-returning splitters."""

    @property
    def page_content(self) -> str:
        ...


class DocumentTextSplitter(Protocol):
    """Minimal interface required from splitters returning documents."""

    def split_text(self, text: str) -> Sequence[SplitDocument]:
        ...


@dataclass(frozen=True)
class TextSplitterChunker:
    """Adapter for library splitters whose split_text() returns strings."""

    splitter: StringTextSplitter

    def chunk(self, document: Document) -> list[Chunk]:
        chunk_texts = self.splitter.split_text(document.text)
        return _build_chunks(
            document_id=document.document_id,
            source_text=document.text,
            chunk_texts=chunk_texts,
        )


@dataclass(frozen=True)
class MarkdownChunker:
    """Adapter for a Markdown splitter that returns document-like objects."""

    splitter: DocumentTextSplitter

    def chunk(self, document: Document) -> list[Chunk]:
        split_documents = self.splitter.split_text(document.text)
        chunk_texts = [split.page_content for split in split_documents]

        return _build_chunks(
            document_id=document.document_id,
            source_text=document.text,
            chunk_texts=chunk_texts,
        )


@dataclass(frozen=True)
class MarkdownRecursiveChunker:
    """Split Markdown by structure, then recursively split each section."""

    markdown_splitter: DocumentTextSplitter
    recursive_splitter: StringTextSplitter

    def chunk(self, document: Document) -> list[Chunk]:
        section_documents = self.markdown_splitter.split_text(document.text)
        section_texts = [section.page_content for section in section_documents]

        section_spans = _locate_chunk_spans(
            source_text=document.text,
            chunk_texts=section_texts,
        )

        chunks: list[Chunk] = []

        for section_text, section_start, _ in section_spans:
            subchunk_texts = self.recursive_splitter.split_text(section_text)

            chunks.extend(
                _build_chunks(
                    document_id=document.document_id,
                    source_text=section_text,
                    chunk_texts=subchunk_texts,
                    base_offset=section_start,
                )
            )

        return chunks


def _build_chunks(
    *,
    document_id: str,
    source_text: str,
    chunk_texts: Sequence[str],
    base_offset: int = 0,
) -> list[Chunk]:
    """Convert exact source substrings into project Chunk objects."""

    spans = _locate_chunk_spans(
        source_text=source_text,
        chunk_texts=chunk_texts,
    )

    chunks: list[Chunk] = []

    for chunk_text, local_start, local_end in spans:
        start_offset = base_offset + local_start
        end_offset = base_offset + local_end

        chunks.append(
            Chunk(
                chunk_id=_make_chunk_id(
                    document_id=document_id,
                    start_offset=start_offset,
                    end_offset=end_offset,
                ),
                document_id=document_id,
                text=chunk_text,
                start_offset=start_offset,
                end_offset=end_offset,
            )
        )

    return chunks


def _locate_chunk_spans(
    *,
    source_text: str,
    chunk_texts: Sequence[str],
) -> list[tuple[str, int, int]]:
    """Locate ordered chunk strings as exact half-open spans in source_text."""

    spans: list[tuple[str, int, int]] = []

    previous_start: int | None = None
    previous_end: int | None = None

    for chunk_text in chunk_texts:
        if not chunk_text:
            continue

        if previous_start is None or previous_end is None:
            search_from = 0
        else:
            # A non-redundant next span that progresses through the source
            # cannot start earlier than previous_end - len(chunk_text).
            # + previous_start + 1 prevents mapping two chunks to the same span.
            search_from = max(
                previous_start + 1,
                previous_end - len(chunk_text),
                0,
            )

        start = source_text.find(chunk_text, search_from)

        if start < 0:
            raise ValueError(
                "Chunk text cannot be mapped exactly to the source document. "
            )

        end = start + len(chunk_text)

        if source_text[start:end] != chunk_text:
            raise ValueError(
                "Invalid chunk offsets: source slice does not equal chunk text."
            )

        spans.append((chunk_text, start, end))
        previous_start = start
        previous_end = end

    return spans


def _make_chunk_id(
    *,
    document_id: str,
    start_offset: int,
    end_offset: int,
) -> str:
    """Create a deterministic chunk identifier from its source span."""

    return f"{document_id}:{start_offset}:{end_offset}"
