from __future__ import annotations

from collections.abc import Sequence
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from rag.models import Chunk


class QdrantStore:

    def __init__(
        self,
        client: QdrantClient,
        *,
        upsert_batch_size: int = 256,
    ) -> None:
        if upsert_batch_size <= 0:
            raise ValueError(
                "upsert_batch_size must be greater than 0"
            )

        self._client = client
        self._upsert_batch_size = upsert_batch_size

    def create_collection(
        self,
        collection_name: str,
        *,
        vector_size: int,
        recreate: bool = False,
    ) -> None:

        if not collection_name:
            raise ValueError(
                "collection_name must not be empty"
            )

        if vector_size <= 0:
            raise ValueError(
                "vector_size must be greater than 0"
            )

        exists = self._client.collection_exists(
            collection_name=collection_name,
        )

        if exists:
            if not recreate:
                raise ValueError(
                    f"Collection '{collection_name}' already exists"
                )

            self._client.delete_collection(
                collection_name=collection_name,
            )

        self._client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

    def delete_collection(
        self,
        collection_name: str,
    ) -> None:

        if self._client.collection_exists(
            collection_name=collection_name,
        ):
            self._client.delete_collection(
                collection_name=collection_name,
            )

    def upsert_chunks(
        self,
        collection_name: str,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:

        if len(chunks) != len(vectors):
            raise ValueError(
                "Number of chunks must match number of vectors"
            )

        if len(chunks) == 0:
            return

        for start in range(
            0,
            len(chunks),
            self._upsert_batch_size,
        ):
            end = start + self._upsert_batch_size

            chunk_batch = chunks[start:end]
            vector_batch = vectors[start:end]

            points = [
                self._to_point(
                    chunk=chunk,
                    vector=vector,
                )
                for chunk, vector in zip(
                    chunk_batch,
                    vector_batch,
                )
            ]

            self._client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True,
            )

    def search(
        self,
        collection_name: str,
        query_vector: Sequence[float],
        *,
        limit: int,
    ) -> list[Chunk]:

        if limit <= 0:
            raise ValueError(
                "limit must be greater than 0"
            )

        if len(query_vector) == 0:
            raise ValueError(
                "query_vector must not be empty"
                )

        result = self._client.query_points(
            collection_name=collection_name,
            query=[
                float(value)
                for value in query_vector
            ],
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        return [
            self._chunk_from_payload(point.payload)
            for point in result.points
        ]

    @staticmethod
    def _to_point(
        *,
        chunk: Chunk,
        vector: Sequence[float],
    ) -> PointStruct:
        if len(vector) == 0:
            raise ValueError(
                f"Vector for chunk '{chunk.chunk_id}' must not be empty"
            )

        return PointStruct(
            id=_chunk_point_id(chunk.chunk_id),
            vector=[
                float(value)
                for value in vector
            ],
            payload={
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "start_offset": chunk.start_offset,
                "end_offset": chunk.end_offset,
            },
        )

    @staticmethod
    def _chunk_from_payload(
        payload: dict | None,
    ) -> Chunk:
        if payload is None:
            raise ValueError(
                "Qdrant search result has no payload"
            )

        required_fields = (
            "chunk_id",
            "document_id",
            "text",
            "start_offset",
            "end_offset",
        )

        missing_fields = [
            field
            for field in required_fields
            if field not in payload
        ]

        if missing_fields:
            raise ValueError(
                "Qdrant payload is missing required fields: "
                + ", ".join(missing_fields)
            )

        return Chunk(
            chunk_id=str(payload["chunk_id"]),
            document_id=str(payload["document_id"]),
            text=str(payload["text"]),
            start_offset=int(payload["start_offset"]),
            end_offset=int(payload["end_offset"]),
        )


def _chunk_point_id(
    chunk_id: str,
) -> str:

    return str(
        uuid5(
            NAMESPACE_URL,
            f"rag-chunk:{chunk_id}",
        )
    )