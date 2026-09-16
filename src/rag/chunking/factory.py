from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

from langchain_text_splitters import (
    NLTKTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_text_splitters.markdown import (
    ExperimentalMarkdownSyntaxTextSplitter,
)

from rag.chunking.chunker import (
    Chunker,
    MarkdownChunker,
    MarkdownRecursiveChunker,
    TextSplitterChunker,
)


ChunkingStrategy = Literal[
    "recursive",
    "sentence",
    "markdown",
    "markdown_recursive",
]

DEFAULT_MARKDOWN_HEADERS: tuple[tuple[str, str], ...] = (
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
    ("#####", "Header 5"),
    ("######", "Header 6"),
)


def create_chunker(
    strategy: ChunkingStrategy,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    length_function: Callable[[str], int] = len,
    sentence_language: str = "russian",
    markdown_headers: Sequence[tuple[str, str]] = DEFAULT_MARKDOWN_HEADERS,
) -> Chunker:

    if strategy == "markdown":
        splitter = _create_markdown_splitter(
            headers=markdown_headers,
        )
        return MarkdownChunker(splitter=splitter)

    if strategy == "recursive":

        chunk_size, chunk_overlap = _require_size_parameters(
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        splitter = _create_recursive_splitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
        )
        return TextSplitterChunker(splitter=splitter)

    if strategy == "sentence":

        chunk_size, chunk_overlap = _require_size_parameters(
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        splitter = _create_sentence_splitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            language=sentence_language,
        )
        return TextSplitterChunker(splitter=splitter)

    if strategy == "markdown_recursive":

        chunk_size, chunk_overlap = _require_size_parameters(
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        markdown_splitter = _create_markdown_splitter(
            headers=markdown_headers,
        )
        recursive_splitter = _create_recursive_splitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
        )

        return MarkdownRecursiveChunker(
            markdown_splitter=markdown_splitter,
            recursive_splitter=recursive_splitter,
        )

    raise ValueError(
        f"Unsupported chunking strategy: {strategy}"
    )


def _create_recursive_splitter(
    *,
    chunk_size: int,
    chunk_overlap: int,
    length_function: Callable[[str], int],
) -> RecursiveCharacterTextSplitter:

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_function,
        keep_separator=True,
        strip_whitespace=False,
    )


def _create_sentence_splitter(
    *,
    chunk_size: int,
    chunk_overlap: int,
    length_function: Callable[[str], int],
    language: str,
) -> NLTKTextSplitter:

    return NLTKTextSplitter(
        separator="",
        language=language,
        use_span_tokenize=True,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_function,
        strip_whitespace=False,
    )


def _create_markdown_splitter(
    *,
    headers: Sequence[tuple[str, str]],
) -> ExperimentalMarkdownSyntaxTextSplitter:

    return ExperimentalMarkdownSyntaxTextSplitter(
        headers_to_split_on=list(headers),
        strip_headers=False,
    )


def _require_size_parameters(
    *,
    strategy: ChunkingStrategy,
    chunk_size: int | None,
    chunk_overlap: int | None,
) -> tuple[int, int]:

    if chunk_size is None:
        raise ValueError(
            f"chunk_size is required for strategy '{strategy}'."
        )

    if chunk_overlap is None:
        raise ValueError(
            f"chunk_overlap is required for strategy '{strategy}'."
        )

    _validate_size_parameters(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return chunk_size, chunk_overlap


def _validate_size_parameters(
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> None:

    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise ValueError(
            "chunk_size must be an integer"
        )

    if isinstance(chunk_overlap, bool) or not isinstance(
        chunk_overlap,
        int,
    ):
        raise ValueError(
            "chunk_overlap must be an integer"
        )

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0"
        )

    if chunk_overlap < 0:
        raise ValueError(
            "chunk_overlap must be >= 0"
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )