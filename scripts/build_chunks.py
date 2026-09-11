from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag.chunking.factory import ChunkingStrategy, create_chunker
from rag.loader import load_documents
from rag.models import Chunk, Document


def build_chunks(
    documents: list[Document],
    *,
    strategy: ChunkingStrategy,
    chunk_size: int,
    chunk_overlap: int,
    sentence_language: str = "russian",
) -> list[Chunk]:
    """Build chunks for all documents using the selected strategy."""

    chunker = create_chunker(
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        sentence_language=sentence_language,
    )

    chunks: list[Chunk] = []

    for document in documents:
        document_chunks = chunker.chunk(document)

        for chunk in document_chunks:
            _validate_chunk(
                document=document,
                chunk=chunk,
            )

        chunks.extend(document_chunks)

    _validate_unique_chunk_ids(chunks)

    return chunks


def save_chunks(
    chunks: list[Chunk],
    output_path: str | Path,
) -> None:
    """Save chunks to a JSONL file."""

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            record = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "start_offset": chunk.start_offset,
                "end_offset": chunk.end_offset,
            }

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
            )
            file.write("\n")


def _validate_chunk(
    *,
    document: Document,
    chunk: Chunk,
) -> None:
    """Validate chunk metadata and source offsets."""

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
    chunks: list[Chunk],
) -> None:
    """Validate that chunk identifiers are unique."""

    seen_ids: set[str] = set()

    for chunk in chunks:
        if chunk.chunk_id in seen_ids:
            raise ValueError(
                f"Duplicate chunk_id: {chunk.chunk_id}"
            )

        seen_ids.add(chunk.chunk_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build chunks from prepared RAG documents."
    )

    parser.add_argument(
        "--prepared-dir",
        type=Path,
        required=True,
        help="Directory containing prepared Markdown documents.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the output JSONL file.",
    )

    parser.add_argument(
        "--strategy",
        choices=(
            "recursive",
            "sentence",
            "markdown",
            "markdown_recursive",
        ),
        required=True,
        help="Chunking strategy.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help=(
            "Maximum chunk size in units used by the splitter's "
            "length function."
        ),
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=64,
        help=(
            "Chunk overlap in units used by the splitter's "
            "length function."
        ),
    )

    parser.add_argument(
        "--sentence-language",
        type=str,
        default="russian",
        help="Language used by the sentence splitter.",
    )

    return parser.parse_args()


def _ensure_nltk_sentence_data(
    language: str,
) -> None:
    import nltk

    resource_path = f"tokenizers/punkt_tab/{language}/"

    try:
        nltk.data.find(resource_path)
    except LookupError:
        print(
            f"NLTK sentence tokenizer data for '{language}' "
            "not found. Downloading 'punkt_tab'..."
        )

        success = nltk.download(
            "punkt_tab",
            quiet=True,
        )

        if not success:
            raise RuntimeError(
                "Failed to download NLTK resource 'punkt_tab'."
            )

        try:
            nltk.data.find(resource_path)
        except LookupError as exc:
            raise RuntimeError(
                f"NLTK resource 'punkt_tab' was downloaded, "
                f"but language '{language}' is still unavailable."
            ) from exc


def main() -> None:
    args = parse_args()

    if args.strategy == "sentence":
        _ensure_nltk_sentence_data(
            language=args.sentence_language,
        )

    documents = load_documents(args.prepared_dir)

    chunks = build_chunks(
        documents=documents,
        strategy=args.strategy,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        sentence_language=args.sentence_language,
    )

    save_chunks(
        chunks=chunks,
        output_path=args.output,
    )

    print(
        f"Built {len(chunks)} chunks "
        f"from {len(documents)} documents "
        f"using '{args.strategy}' strategy."
    )
    print(f"Saved chunks to: {args.output}")


if __name__ == "__main__":
    main()