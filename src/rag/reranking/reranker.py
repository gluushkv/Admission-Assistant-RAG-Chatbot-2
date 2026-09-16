from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sentence_transformers import CrossEncoder

from rag.models import Chunk


class CrossEncoderReranker:

    def __init__(
        self,
        model_id: str,
        *,
        device: str = "cuda",
        batch_size: int = 16,
        trust_remote_code: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than 0"
            )

        self._model_id = model_id
        self._batch_size = batch_size

        self._model = CrossEncoder(
            model_id,
            device=device,
            trust_remote_code=trust_remote_code,
        )

    @property
    def model_id(self) -> str:
        return self._model_id

    def rerank(
        self,
        query: str,
        chunks: Sequence[Chunk],
        *,
        top_k: int | None = None,
    ) -> list[Chunk]:
        if not query.strip():
            raise ValueError(
                "query must not be empty"
            )

        if len(chunks) == 0:
            return []

        if top_k is not None and top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0"
            )

        pairs = [
            (query, chunk.text)
            for chunk in chunks
        ]

        scores = self._model.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )

        scores = np.asarray(
            scores,
            dtype=np.float64,
        ).reshape(-1)

        if len(scores) != len(chunks):
            raise RuntimeError(
                "Number of reranker scores does not "
                "match number of chunks"
            )

        ranked_indices = np.argsort(
            -scores,
            kind="stable",
        )

        if top_k is not None:
            ranked_indices = ranked_indices[
                :top_k
            ]

        return [
            chunks[int(index)]
            for index in ranked_indices
        ]