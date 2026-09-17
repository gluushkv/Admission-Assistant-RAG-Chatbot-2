from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import yaml

from rag.models import Chunk


def format_answer_with_sources(
    *,
    answer: str,
    used_fragment_indices: Sequence[int],
    chunks: Sequence[Chunk],
    documents_config_path: str | Path,
) -> str:
    used_chunks = _select_used_chunks(
        chunks=chunks,
        used_fragment_indices=used_fragment_indices,
    )

    sources = _collect_sources(
        chunks=used_chunks,
        documents_config_path=documents_config_path,
    )

    if len(sources) == 0:
        return answer.strip()

    sources_text = "\n".join(
        f"- {source['title']}: {source['url']}"
        for source in sources
    )

    return (
        f"{answer.strip()}\n\n"
        f"Обратите внимание, что бот может допускать ошибки "
        f"или неточности. Рекомендуем перепроверять важную "
        f"информацию в официальных источниках.\n\n"
        f"Источники:\n"
        f"{sources_text}"
    )


def _select_used_chunks(
    *,
    chunks: Sequence[Chunk],
    used_fragment_indices: Sequence[int],
) -> list[Chunk]:
    used_chunks: list[Chunk] = []
    seen_indices: set[int] = set()

    for fragment_index in used_fragment_indices:
        if fragment_index in seen_indices:
            continue

        if not 1 <= fragment_index <= len(chunks):
            raise ValueError(
                "Fragment index is out of range: "
                f"{fragment_index}"
            )

        seen_indices.add(
            fragment_index
        )

        used_chunks.append(
            chunks[fragment_index - 1]
        )

    return used_chunks


def _collect_sources(
    *,
    chunks: Sequence[Chunk],
    documents_config_path: str | Path,
) -> list[dict[str, str]]:
    document_sources = _load_document_sources(
        documents_config_path
    )

    sources: list[dict[str, str]] = []
    seen_document_ids: set[str] = set()

    for chunk in chunks:
        document_id = chunk.document_id

        if document_id in seen_document_ids:
            continue

        source = document_sources.get(
            document_id
        )

        if source is None:
            continue

        seen_document_ids.add(
            document_id
        )

        sources.append(
            source
        )

    return sources


def _load_document_sources(
    path: str | Path,
) -> dict[str, dict[str, str]]:
    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(
            file
        )

    if not isinstance(config, dict):
        raise ValueError(
            "documents.yaml must contain "
            "a mapping"
        )

    documents = config.get(
        "documents"
    )

    if not isinstance(documents, dict):
        raise ValueError(
            "documents.yaml must contain "
            "'documents' mapping"
        )

    sources: dict[
        str,
        dict[str, str],
    ] = {}

    for document_id, document in documents.items():
        if not isinstance(document, dict):
            continue

        title = document.get(
            "title"
        )
        url = document.get(
            "url"
        )

        if not title or not url:
            continue

        sources[str(document_id)] = {
            "title": str(title),
            "url": str(url),
        }

    return sources