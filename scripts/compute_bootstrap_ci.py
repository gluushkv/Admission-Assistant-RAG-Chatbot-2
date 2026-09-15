from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag.evaluation.significance import (
    paired_bootstrap_difference_ci,
)


def load_records(
    path: str | Path,
) -> list[dict]:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Results file not found: {path}"
        )

    records: list[dict] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at "
                    f"{path}:{line_number}"
                ) from exc

            records.append(record)

    if not records:
        raise ValueError(
            f"Results file is empty: {path}"
        )

    return records


def extract_scores(
    records: list[dict],
    *,
    configuration: str,
    metric_group: str,
    metric_name: str,
    evidence_relation: str | None = None,
) -> dict[str, float]:
    scores: dict[str, float] = {}

    for record in records:
        if record["configuration"] != configuration:
            continue

        if (
            evidence_relation is not None
            and record.get("evidence_relation")
            != evidence_relation
        ):
            continue

        question_id = record["question_id"]

        if question_id in scores:
            raise ValueError(
                f"Duplicate result for question "
                f"'{question_id}' in configuration "
                f"'{configuration}'."
            )

        try:
            value = record[
                metric_group
            ][metric_name]
        except KeyError as exc:
            raise ValueError(
                f"Metric '{metric_group}.{metric_name}' "
                "is missing from a result record."
            ) from exc

        scores[question_id] = float(value)

    if not scores:
        raise ValueError(
            f"No scores found for configuration "
            f"'{configuration}'."
        )

    return scores


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute a paired bootstrap confidence "
            "interval for the difference between "
            "two experiment configurations."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to raw results JSONL.",
    )

    parser.add_argument(
        "--a",
        required=True,
        help="Baseline configuration.",
    )

    parser.add_argument(
        "--b",
        required=True,
        help="Compared configuration.",
    )

    parser.add_argument(
        "--metric-group",
        required=True,
        choices=[
            "single_chunk_metrics",
            "context_metrics",
        ],
    )

    parser.add_argument(
        "--metric",
        required=True,
        help="Metric name, for example mrr@5.",
    )

    parser.add_argument(
        "--evidence-relation",
        choices=[
            "AND",
            "OR",
        ],
        default=None,
        help=(
            "Optionally restrict comparison to "
            "one evidence relation."
        ),
    )

    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--n-resamples",
        type=int,
        default=10_000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    records = load_records(args.input)

    scores_a = extract_scores(
        records,
        configuration=args.a,
        metric_group=args.metric_group,
        metric_name=args.metric,
        evidence_relation=args.evidence_relation,
    )

    scores_b = extract_scores(
        records,
        configuration=args.b,
        metric_group=args.metric_group,
        metric_name=args.metric,
        evidence_relation=args.evidence_relation,
    )

    result = paired_bootstrap_difference_ci(
        scores_a,
        scores_b,
        confidence_level=args.confidence_level,
        n_resamples=args.n_resamples,
        seed=args.seed,
    )

    print(
        f"Comparison: {args.b} - {args.a}"
    )
    print(
        f"Metric: "
        f"{args.metric_group}.{args.metric}"
    )
    print(
        f"N: {result.n_items}"
    )
    print(
        f"Mean A: {result.mean_a:.6f}"
    )
    print(
        f"Mean B: {result.mean_b:.6f}"
    )
    print(
        f"Difference B-A: "
        f"{result.mean_difference:.6f}"
    )
    print(
        f"{result.confidence_level:.0%} CI: "
        f"[{result.lower:.6f}, "
        f"{result.upper:.6f}]"
    )
    print(
        f"Contains zero: "
        f"{result.contains_zero}"
    )


if __name__ == "__main__":
    main()