from __future__ import annotations

import csv
import gc
import json
from collections.abc import Sequence
from pathlib import Path

import torch
import yaml
from huggingface_hub import (
    snapshot_download,
)

from experiments.embeddings.efficiency import (
    measure_embedding_efficiency,
)
from experiments.embeddings.retrieval_quality import (
    build_retrieval_quality_summary,
    run_retrieval_quality,
)
from rag.embeddings.factory import (
    create_embedder,
)
from rag.evaluation.models import (
    GoldenQuestion,
)
from rag.models import Chunk
from rag.retrieval.qdrant_store import (
    QdrantStore,
)


def run_embeddings_experiment(
    *,
    chunks: Sequence[Chunk],
    questions: Sequence[GoldenQuestion],
    store: QdrantStore,
    config_path: str | Path = (
        "configs/embeddings.yaml"
    ),
    output_dir: str | Path = (
        "results/embeddings"
    ),
    device: str = "cuda",
) -> list[dict]:


    if len(chunks) == 0:
        raise ValueError(
            "chunks must not be empty"
        )

    if len(questions) == 0:
        raise ValueError(
            "questions must not be empty"
        )

    cuda_device = torch.device(
        device
    )

    if cuda_device.type != "cuda":
        raise ValueError(
            "Embedding experiment is GPU-only"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available"
        )

    config = _load_config(
        config_path
    )

    common = config[
        "common"
    ]

    model_configs = config[
        "models"
    ]

    ks = tuple(
        int(k)
        for k in common.get(
            "ks",
            (1, 3, 5),
        )
    )

    if len(ks) == 0:
        raise ValueError(
            "ks must not be empty"
        )

    if any(k <= 0 for k in ks):
        raise ValueError(
            "All k values must be greater than 0"
        )

    batch_size = int(
        common.get(
            "batch_size",
            32,
        )
    )

    warmup_documents = int(
        common.get(
            "warmup_documents",
            32,
        )
    )

    warmup_queries = int(
        common.get(
            "warmup_queries",
            5,
        )
    )

    query_repeats = int(
        common.get(
            "query_repeats",
            1,
        )
    )

    corpus_texts = [
        chunk.text
        for chunk in chunks
    ]

    query_texts = [
        question.question
        for question in questions
        if question.is_answerable
    ]

    if len(query_texts) == 0:
        raise ValueError(
            "No answerable questions were provided"
        )

    results: list[dict] = []

    for (
        configuration_name,
        model_config,
    ) in model_configs.items():

        print()
        print(
            f"Running embedding model: "
            f"{configuration_name}"
        )

        model_id = str(
            model_config[
                "model_id"
            ]
        )

        print(
            f"Model ID: {model_id}"
        )

        model_batch_size = int(
            model_config.get(
                "batch_size",
                batch_size,
            )
        )

        embedder = create_embedder(
            model_id=model_id,
            device=device,
            batch_size=(
                model_batch_size
            ),
            query_prefix=(
                model_config.get(
                    "query_prefix",
                    "",
                )
            ),
            document_prefix=(
                model_config.get(
                    "document_prefix",
                    "",
                )
            ),
            query_instruction=(
                model_config.get(
                    "query_instruction"
                )
            ),
        )

        try:

            (
                vectors,
                efficiency,
            ) = (
                measure_embedding_efficiency(
                    embedder=embedder,
                    corpus_texts=corpus_texts,
                    queries=query_texts,
                    device=device,
                    warmup_documents=(
                        warmup_documents
                    ),
                    warmup_queries=(
                        warmup_queries
                    ),
                    query_repeats=(
                        query_repeats
                    ),
                )
            )

            efficiency[
                "batch_size"
            ] = model_batch_size

            efficiency[
                "model_checkpoint_size_bytes"
            ] = (
                _get_model_checkpoint_size_bytes(
                    model_id
                )
            )

            records = (
                run_retrieval_quality(
                    chunks=chunks,
                    vectors=vectors,
                    questions=questions,
                    embedder=embedder,
                    store=store,
                    configuration_name=(
                        configuration_name
                    ),
                    ks=ks,
                    collection_name=(
                        "embedding-experiment"
                    ),
                )
            )

            quality_summary = (
                build_retrieval_quality_summary(
                    records
                )
            )

            result = {
                "configuration": (
                    configuration_name
                ),
                "model_id": (
                    embedder.model_id
                ),
                "records": (
                    records
                ),
                "quality_summary": (
                    quality_summary
                ),
                "efficiency": {
                    "configuration": (
                        configuration_name
                    ),
                    "model_id": (
                        embedder.model_id
                    ),
                    **efficiency,
                },
            }

            results.append(
                result
            )

            _print_result(
                result
            )

            _save_results(
                results=results,
                output_dir=output_dir,
            )

        finally:
            del embedder

            gc.collect()

            torch.cuda.empty_cache()

    return results


def _load_config(
    path: str | Path,
) -> dict:

    path = Path(
        path
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Embedding config not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(
            file
        )

    if not isinstance(
        config,
        dict,
    ):
        raise ValueError(
            "Embedding config must contain "
            "a top-level mapping"
        )

    common = config.get(
        "common"
    )

    models = config.get(
        "models"
    )

    if not isinstance(
        common,
        dict,
    ):
        raise ValueError(
            "Embedding config must contain "
            "'common' mapping"
        )

    if not isinstance(
        models,
        dict,
    ):
        raise ValueError(
            "Embedding config must contain "
            "'models' mapping"
        )

    if len(models) == 0:
        raise ValueError(
            "'models' must not be empty"
        )

    return config


def _get_model_checkpoint_size_bytes(
    model_id: str,
) -> int:

    model_path = Path(
        model_id
    ).expanduser()

    if model_path.exists():
        return _path_size_bytes(
            model_path
        )

    snapshot_path = Path(
        snapshot_download(
            repo_id=model_id,
            local_files_only=True,
        )
    )

    return _path_size_bytes(
        snapshot_path
    )


def _path_size_bytes(
    path: Path,
) -> int:

    if path.is_file():
        return int(
            path.stat().st_size
        )

    if not path.is_dir():
        raise FileNotFoundError(
            f"Model path does not exist: {path}"
        )

    total_bytes = 0

    seen_files: set[
        tuple[int, int]
    ] = set()

    for file_path in path.rglob(
        "*"
    ):
        if not file_path.is_file():
            continue

        stat = file_path.stat()

        file_key = (
            int(stat.st_dev),
            int(stat.st_ino),
        )

        if file_key in seen_files:
            continue

        seen_files.add(
            file_key
        )

        total_bytes += int(
            stat.st_size
        )

    if total_bytes <= 0:
        raise RuntimeError(
            f"Model checkpoint is empty: {path}"
        )

    return total_bytes


def _save_results(
    *,
    results: Sequence[dict],
    output_dir: str | Path,
) -> None:

    output_dir = Path(
        output_dir
    )

    raw_dir = (
        output_dir
        / "raw"
    )

    tables_dir = (
        output_dir
        / "tables"
    )

    raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tables_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    _save_retrieval_records(
        results=results,
        path=(
            raw_dir
            / "retrieval_results.jsonl"
        ),
    )

    _save_efficiency_records(
        results=results,
        path=(
            raw_dir
            / "efficiency.jsonl"
        ),
    )

    _save_quality_table(
        results=results,
        path=(
            tables_dir
            / "retrieval_quality.csv"
        ),
    )

    _save_quality_efficiency_table(
        results=results,
        path=(
            tables_dir
            / "quality_efficiency.csv"
        ),
    )


def _save_retrieval_records(
    *,
    results: Sequence[dict],
    path: Path,
) -> None:

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for result in results:

            for record in result[
                "records"
            ]:

                file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                )

                file.write(
                    "\n"
                )


def _save_efficiency_records(
    *,
    results: Sequence[dict],
    path: Path,
) -> None:

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for result in results:

            file.write(
                json.dumps(
                    result[
                        "efficiency"
                    ],
                    ensure_ascii=False,
                )
            )

            file.write(
                "\n"
            )


def _save_quality_table(
    *,
    results: Sequence[dict],
    path: Path,
) -> None:

    rows: list[dict] = []

    for result in results:

        for quality_row in result[
            "quality_summary"
        ]:

            rows.append(
                dict(
                    quality_row
                )
            )

    _write_csv(
        rows=rows,
        path=path,
    )


def _save_quality_efficiency_table(
    *,
    results: Sequence[dict],
    path: Path,
) -> None:

    rows: list[dict] = []

    for result in results:

        efficiency = result[
            "efficiency"
        ]

        for quality_row in result[
            "quality_summary"
        ]:

            row = {
                **quality_row,
                "embedding_dimension": (
                    efficiency[
                        "embedding_dimension"
                    ]
                ),
                "corpus_embedding_time_seconds": (
                    efficiency[
                        "corpus_embedding_time_seconds"
                    ]
                ),
                "corpus_embedding_throughput_chunks_per_sec": (
                    efficiency[
                        "corpus_embedding_throughput_chunks_per_sec"
                    ]
                ),
                "median_query_embedding_latency_ms": (
                    efficiency[
                        "median_query_embedding_latency_ms"
                    ]
                ),
                "p95_query_embedding_latency_ms": (
                    efficiency[
                        "p95_query_embedding_latency_ms"
                    ]
                ),
                "corpus_peak_vram_bytes": (
                    efficiency[
                        "corpus_peak_vram_bytes"
                    ]
                ),
                "query_peak_vram_bytes": (
                    efficiency[
                        "query_peak_vram_bytes"
                    ]
                ),
                "raw_vector_size_bytes": (
                    efficiency[
                        "raw_vector_size_bytes"
                    ]
                ),
                "model_checkpoint_size_bytes": (
                    efficiency[
                        "model_checkpoint_size_bytes"
                    ]
                ),
                "batch_size": (
                    efficiency[
                        "batch_size"
                    ]
                ),
            }

            rows.append(
                row
            )

    _write_csv(
        rows=rows,
        path=path,
    )


def _write_csv(
    *,
    rows: Sequence[dict],
    path: Path,
) -> None:

    if len(rows) == 0:
        raise ValueError(
            "Cannot save an empty table"
        )

    fieldnames: list[
        str
    ] = []

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(
                    key
                )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def _print_result(
    result: dict,
) -> None:

    efficiency = result[
        "efficiency"
    ]

    all_answerable = next(
        (
            row
            for row in result[
                "quality_summary"
            ]
            if row[
                "question_scope"
            ] == "all_answerable"
        ),
        None,
    )

    print()

    if all_answerable is not None:

        recall_5 = (
            all_answerable.get(
                "single_chunk_recall@5"
            )
        )

        mrr_5 = (
            all_answerable.get(
                "single_chunk_mrr@5"
            )
        )

        if recall_5 is not None:
            print(
                "Recall@5: "
                f"{float(recall_5):.4f}"
            )

        if mrr_5 is not None:
            print(
                "MRR@5: "
                f"{float(mrr_5):.4f}"
            )

    print(
        "Corpus embedding time: "
        f"{efficiency['corpus_embedding_time_seconds']:.3f}s"
    )

    print(
        "Corpus throughput: "
        f"{efficiency['corpus_embedding_throughput_chunks_per_sec']:.2f} "
        "chunks/s"
    )

    print(
        "Median query latency: "
        f"{efficiency['median_query_embedding_latency_ms']:.3f}ms"
    )

    print(
        "P95 query latency: "
        f"{efficiency['p95_query_embedding_latency_ms']:.3f}ms"
    )

    print(
        "Corpus peak VRAM: "
        f"{efficiency['corpus_peak_vram_bytes']} bytes"
    )

    print(
        "Query peak VRAM: "
        f"{efficiency['query_peak_vram_bytes']} bytes"
    )

    print(
        "Embedding dimension: "
        f"{efficiency['embedding_dimension']}"
    )

    print(
        "Raw vector size: "
        f"{efficiency['raw_vector_size_bytes']} bytes"
    )

    print(
        "Model checkpoint size: "
        f"{efficiency['model_checkpoint_size_bytes']} bytes"
    )