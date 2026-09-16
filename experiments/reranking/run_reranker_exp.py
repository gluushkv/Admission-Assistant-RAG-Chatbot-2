from __future__ import annotations

import csv
import gc
import json
from collections.abc import Sequence
from pathlib import Path

import torch
import yaml

from experiments.reranking.efficiency import (
    measure_reranking_efficiency,
)
from experiments.reranking.retrieval_candidates import (
    retrieve_candidates,
)
from experiments.reranking.retrieval_quality import (
    aggregate_retrieval_quality,
    build_baseline_rankings,
    evaluate_rankings,
)
from rag.embeddings.embedder import (
    SentenceTransformerEmbedder,
)
from rag.evaluation.models import GoldenQuestion
from rag.models import Chunk
from rag.reranking.factory import create_reranker
from rag.retrieval.indexing import build_dense_index
from rag.retrieval.qdrant_store import QdrantStore


def run_reranker_experiment(
    *,
    chunks: Sequence[Chunk],
    questions: Sequence[GoldenQuestion],
    embedder: SentenceTransformerEmbedder,
    store: QdrantStore,
    config_path: str | Path = (
        "configs/reranking.yaml"
    ),
    output_dir: str | Path = (
        "results/reranking/reranker"
    ),
    device: str = "cuda",
) -> list[dict]:
    """Compare no-reranker baseline with configured rerankers."""

    if len(chunks) == 0:
        raise ValueError(
            "chunks must not be empty"
        )

    config = _load_config(
        config_path
    )

    common = config["common"]
    experiment = config[
        "reranker_experiment"
    ]

    ks = tuple(
        int(k)
        for k in common.get(
            "ks",
            (1, 3, 5),
        )
    )

    final_k = int(
        common.get(
            "final_k",
            5,
        )
    )

    warmup_queries = int(
        common.get(
            "warmup_queries",
            3,
        )
    )

    default_batch_size = int(
        common.get(
            "batch_size",
            16,
        )
    )

    candidate_k = int(
        experiment[
            "candidate_k"
        ]
    )

    collection_name = (
        "reranker-experiment"
    )


    build_dense_index(
        chunks=chunks,
        embedder=embedder,
        store=store,
        collection_name=collection_name,
        recreate=True,
    )

    candidates = retrieve_candidates(
        questions=questions,
        embedder=embedder,
        store=store,
        collection_name=collection_name,
        candidate_k=candidate_k,
    )

    results: list[dict] = []

    baseline_rankings = (
        build_baseline_rankings(
            candidates,
            final_k=final_k,
        )
    )

    baseline_records = evaluate_rankings(
        questions=questions,
        rankings=baseline_rankings,
        configuration_name="no_reranker",
        ks=ks,
    )

    baseline_quality = (
        aggregate_retrieval_quality(
            baseline_records
        )
    )

    results.append(
        {
            "configuration": (
                "no_reranker"
            ),
            "model_id": None,
            "candidate_k": candidate_k,
            "final_k": final_k,
            "records": baseline_records,
            "quality": baseline_quality,
            "efficiency": {
                "model_id": None,
                "n_queries": len(
                    baseline_records
                ),
                "median_reranking_latency_ms": 0.0,
                "p95_reranking_latency_ms": 0.0,
                "peak_vram_bytes": 0,
                "peak_vram_gb": 0.0,
            },
        }
    )

    _save_results(
        results=results,
        output_dir=output_dir,
    )

    models = experiment["models"]

    for (
        configuration_name,
        model_config,
    ) in models.items():

        print()
        print(
            f"Running reranker: "
            f"{configuration_name}"
        )

        print(
            f"Model ID: "
            f"{model_config['model_id']}"
        )

        reranker = create_reranker(
            model_id=model_config[
                "model_id"
            ],
            device=device,
            batch_size=int(
                model_config.get(
                    "batch_size",
                    default_batch_size,
                )
            ),
            trust_remote_code=bool(
                model_config.get(
                    "trust_remote_code",
                    False,
                )
            ),
        )

        try:
            rankings, efficiency = (
                measure_reranking_efficiency(
                    reranker=reranker,
                    questions=questions,
                    candidates=candidates,
                    final_k=final_k,
                    warmup_queries=(
                        warmup_queries
                    ),
                )
            )

            records = evaluate_rankings(
                questions=questions,
                rankings=rankings,
                configuration_name=(
                    configuration_name
                ),
                ks=ks,
            )

            quality = (
                aggregate_retrieval_quality(
                    records
                )
            )

            results.append(
                {
                    "configuration": (
                        configuration_name
                    ),
                    "model_id": (
                        reranker.model_id
                    ),
                    "candidate_k": (
                        candidate_k
                    ),
                    "final_k": final_k,
                    "records": records,
                    "quality": quality,
                    "efficiency": (
                        efficiency
                    ),
                }
            )

            _print_result(
                results[-1]
            )

            _save_results(
                results=results,
                output_dir=output_dir,
            )

        finally:
            del reranker

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return results


def _load_config(
    path: str | Path,
) -> dict:

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Reranking config not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(
            file
        )

    if not isinstance(config, dict):
        raise ValueError(
            "Reranking config must contain "
            "a top-level mapping"
        )

    if not isinstance(
        config.get("common"),
        dict,
    ):
        raise ValueError(
            "Config is missing 'common'"
        )

    if not isinstance(
        config.get(
            "reranker_experiment"
        ),
        dict,
    ):
        raise ValueError(
            "Config is missing "
            "'reranker_experiment'"
        )

    return config


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

    with (
        raw_dir
        / "retrieval_results.jsonl"
    ).open(
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
                file.write("\n")

    with (
        raw_dir
        / "efficiency.jsonl"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        for result in results:
            record = {
                "configuration": (
                    result[
                        "configuration"
                    ]
                ),
                "model_id": (
                    result[
                        "model_id"
                    ]
                ),
                "candidate_k": (
                    result[
                        "candidate_k"
                    ]
                ),
                **result["efficiency"],
            }

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
            )
            file.write("\n")

    quality_rows = []

    efficiency_rows = []

    for result in results:
        quality = result[
            "quality"
        ]

        efficiency = result[
            "efficiency"
        ]

        quality_rows.append(
            {
                "configuration": (
                    result[
                        "configuration"
                    ]
                ),
                "model_id": (
                    result[
                        "model_id"
                    ]
                ),
                "candidate_k": (
                    result[
                        "candidate_k"
                    ]
                ),
                "n_questions": len(
                    result["records"]
                ),
                **quality,
            }
        )

        efficiency_rows.append(
            {
                "configuration": (
                    result[
                        "configuration"
                    ]
                ),
                "model_id": (
                    result[
                        "model_id"
                    ]
                ),
                "single_chunk_hit@1": (
                    quality.get(
                        "single_chunk_hit@1"
                    )
                ),
                "single_chunk_mrr@5": (
                    quality.get(
                        "single_chunk_mrr@5"
                    )
                ),
                "single_chunk_recall@5": (
                    quality.get(
                        "single_chunk_recall@5"
                    )
                ),
                "median_reranking_latency_ms": (
                    efficiency[
                        "median_reranking_latency_ms"
                    ]
                ),
                "p95_reranking_latency_ms": (
                    efficiency[
                        "p95_reranking_latency_ms"
                    ]
                ),
                "peak_vram_gb": (
                    efficiency[
                        "peak_vram_gb"
                    ]
                ),
            }
        )

    _write_csv(
        rows=quality_rows,
        path=(
            tables_dir
            / "retrieval_quality.csv"
        ),
    )

    _write_csv(
        rows=efficiency_rows,
        path=(
            tables_dir
            / "quality_efficiency.csv"
        ),
    )


def _write_csv(
    *,
    rows: Sequence[dict],
    path: Path,
) -> None:

    if len(rows) == 0:
        return

    fieldnames: list[str] = []

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

    quality = result[
        "quality"
    ]

    efficiency = result[
        "efficiency"
    ]

    print(
        "Hit@1: "
        f"{quality.get('single_chunk_hit@1', float('nan')):.4f}"
    )

    print(
        "MRR@5: "
        f"{quality.get('single_chunk_mrr@5', float('nan')):.4f}"
    )

    print(
        "Recall@5: "
        f"{quality.get('single_chunk_recall@5', float('nan')):.4f}"
    )

    print(
        "Median reranking latency: "
        f"{efficiency['median_reranking_latency_ms']:.3f} ms"
    )

    print(
        "Peak VRAM: "
        f"{efficiency['peak_vram_gb']:.3f} GB"
    )