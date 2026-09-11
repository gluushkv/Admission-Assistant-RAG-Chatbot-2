from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


TextFormatter = Callable[[str], str]


def _identity(text: str) -> str:
    return text


class SentenceTransformerEmbedder:

    def __init__(
        self,
        *,
        model: SentenceTransformer,
        model_id: str,
        batch_size: int = 32,
        query_formatter: TextFormatter = _identity,
        document_formatter: TextFormatter = _identity,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        self._model = model
        self._model_id = model_id
        self._batch_size = batch_size
        self._query_formatter = query_formatter
        self._document_formatter = document_formatter
        self._normalize_embeddings = normalize_embeddings
        self._show_progress_bar = show_progress_bar

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimension(self) -> int:
        dimension = self._model.get_embedding_dimension()

        if dimension is None:
            raise RuntimeError(
                f"Could not determine embedding dimension "
                f"for model '{self._model_id}'."
            )

        return int(dimension)

    def prepare_queries(
        self,
        texts: Sequence[str],
    ) -> list[str]:

        return [
            self._query_formatter(text)
            for text in texts
        ]

    def prepare_documents(
        self,
        texts: Sequence[str],
    ) -> list[str]:

        return [
            self._document_formatter(text)
            for text in texts
        ]

    def embed_queries(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:

        prepared_texts = self.prepare_queries(texts)

        return self._encode(prepared_texts)

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:

        prepared_texts = self.prepare_documents(texts)

        return self._encode(prepared_texts)

    def _encode(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:
        if not texts:
            return np.empty(
                (0, self.dimension),
                dtype=np.float32,
            )

        embeddings = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            show_progress_bar=self._show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize_embeddings,
        )

        array = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        return array
