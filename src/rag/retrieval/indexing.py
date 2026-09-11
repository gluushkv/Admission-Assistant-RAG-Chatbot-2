from collections.abc import Sequence

from rag.embeddings.embedder import SentenceTransformerEmbedder
from rag.models import Chunk
from rag.retrieval.qdrant_store import QdrantStore


def build_dense_index(
    *,
    chunks: Sequence[Chunk],
    embedder: SentenceTransformerEmbedder,
    store: QdrantStore,
    collection_name: str,
    recreate: bool = True,
) -> None:

    if not chunks:
        raise ValueError("chunks must not be empty")

    vectors = embedder.embed_documents(
        [chunk.text for chunk in chunks]
    )

    store.create_collection(
        collection_name=collection_name,
        vector_size=embedder.dimension,
        recreate=recreate,
    )

    store.upsert_chunks(
        collection_name=collection_name,
        chunks=chunks,
        vectors=vectors,
    )