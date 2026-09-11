from __future__ import annotations

from rag.embeddings.embedder import SentenceTransformerEmbedder
from rag.models import Chunk
from rag.retrieval.qdrant_store import QdrantStore


class DenseRetriever:

    def __init__(
        self,
        *,
        embedder: SentenceTransformerEmbedder,
        store: QdrantStore,
        collection_name: str,
    ) -> None:
        if not collection_name:
            raise ValueError(
                "collection_name must not be empty"
            )

        self._embedder = embedder
        self._store = store
        self._collection_name = collection_name

    @property
    def collection_name(self) -> str:
        return self._collection_name

    def retrieve(
        self,
        query: str,
        *,
        k: int = 5,
    ) -> list[Chunk]:

        if not isinstance(query, str):
            raise TypeError(
                "query must be a string"
            )

        if not query.strip():
            raise ValueError(
                "query must not be empty"
            )

        if k <= 0:
            raise ValueError(
                "k must be greater than 0"
            )

        query_embeddings = (
            self._embedder.embed_queries([query])
        )

        query_vector = query_embeddings[0]

        return self._store.search(
            collection_name=self._collection_name,
            query_vector=query_vector,
            limit=k,
        )