from __future__ import annotations

import csv
import gc
import json
from collections.abc import Sequence
from pathlib import Path
from statistics import fmean

import torch
import yaml

from experiments.reranking.efficiency import (
    measure_reranking_efficiency,
)
from experiments.reranking.retrieval_candidates import (
    retrieve_candidates,
    slice_candidates,
)
from experiments.reranking.retrieval_quality import (
    aggregate_retrieval_quality,
    compute_candidate_recall,
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


def run_candidate_pool_experiment(
    *,
    chunks: Sequence[Chunk],
    questions: Sequence[GoldenQuestion],
    embedder: SentenceTransformerEmbedder,
    store: QdrantStore,
    config_path: str | Path = (
        "configs/reranking.yaml"
    ),
    output_dir: str | Path = (
        "results/reranking/candidate_pool"
    ),
    device: str = "cuda",
) -> list[dict]:

    config = _load_config(
        config_path
    )

    common = config["common"]
    reranker_config = config[
        "reranker_experiment"
    ]
    experiment = config[
        "candidate_pool_experiment"
    ]

    final_k = int(
        common.get(
            "final_k",
            5,
        )
    )

    ks = tuple(
        int(k)
        for k in common.get(
            "ks",
            (1, 3, 5),
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

    candidate_ks = sorted(
        {
            int(value)
            for value in experiment[
                "candidate_ks"
            ]
        }
    )

    if len(candidate_ks) == 0:
        raise ValueError(
            "candidate_ks must not be empty"
        )

    if any(
        candidate_k < final_k
        for candidate_k in candidate_ks
    ):
        raise ValueError(
            "candidate_k must be >= final_k"
        )

    reranker_name = experiment[
        "reranker"
    ]

    models = reranker_config[
        "models"
    ]

    if reranker_name not in models:
        raise ValueError(
            f"Unknown reranker: "
            f"{reranker_name}"
        )

    model_config = models[
        reranker_name
    ]

    collection_name = (
        "candidate-pool-experiment"
    )


    build_dense_index(
        chunks=chunks,
        embedder=embedder,
        store=store,
        collection_name=collection_name,
        recreate=True,
    )

    max_candidate_k = max(
        candidate_ks
    )

    all_candidates = (
        retrieve_candidates(
            questions=questions,
            embedder=embedder,
            store=store,
            collection_name=collection_name,
            candidate_k=max_candidate_k,
        )
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

    results: list[dict] = []

    try:
        for candidate_k in candidate_ks:
            print()
            print(
                f"Candidate pool: "
                f"{candidate_k}"
            )

            candidates = slice_candidates(
                all_candidates,
                candidate_k=candidate_k,
            )

            candidate_recall = (
                compute_candidate_recall(
                    questions=questions,
                    candidates=candidates,
                    candidate_k=(
                        candidate_k
                    ),
                )
            )

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
                    f"{reranker_name}"
                    f"-candidate-{candidate_k}"
                ),
                ks=ks,
            )

            for record in records:
                question_id = record[
                    "question_id"
                ]

                record[
                    "candidate_k"
                ] = candidate_k

                record[
                    "candidate_recall"
                ] = candidate_recall[
                    question_id
                ]

            quality = (
                aggregate_retrieval_quality(
                    records
                )
            )

            mean_candidate_recall = (
                fmean(
                    candidate_recall.values()
                )
            )

            results.append(
                {
                    "configuration": (
                        f"{reranker_name}"
                        f"-candidate-{candidate_k}"
                    ),
                    "reranker": (
                        reranker_name
                    ),
                    "model_id": (
                        reranker.model_id
                    ),
                    "candidate_k": (
                        candidate_k
                    ),
                    "final_k": final_k,
                    "candidate_recall": (
                        mean_candidate_recall
                    ),
                    "records": records,
                    "quality": quality,
                    "efficiency": efficiency,
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

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(
            file
        )

    if not isinstance(config, dict):
        raise ValueError(
            "Invalid reranking configuration"
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

    rows: list[dict] = []

    for result in results:
        quality = result[
            "quality"
        ]

        efficiency = result[
            "efficiency"
        ]

        rows.append(
            {
                "candidate_k": (
                    result[
                        "candidate_k"
                    ]
                ),
                "candidate_recall": (
                    result[
                        "candidate_recall"
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
                "context_recall@5": (
                    quality.get(
                        "context_recall@5"
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
                "peak_vram_bytes": (
                    efficiency[
                        "peak_vram_bytes"
                    ]
                ),
            }
        )

    _write_csv(
        rows=rows,
        path=(
            tables_dir
            / "candidate_pool.csv"
        ),
    )


def _write_csv(
    *,
    rows: Sequence[dict],
    path: Path,
) -> None:

    if len(rows) == 0:
        return

    fieldnames = list(
        rows[0].keys()
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
        "Candidate Recall: "
        f"{result['candidate_recall']:.4f}"
    )

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
        "Median latency: "
        f"{efficiency['median_reranking_latency_ms']:.3f} ms"
    )