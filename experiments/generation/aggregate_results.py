from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import fmean, median

from rag.evaluation.generation_evaluator import (
    evaluate_generation_sample,
)
from rag.evaluation.generation_metrics import (
    correct_abstention_rate,
    false_abstention_rate,
)
from rag.evaluation.models import GoldenQuestion


def aggregate_generation_results(
    *,
    raw_results_path: str | Path,
    annotations_path: str | Path,
    questions: list[GoldenQuestion],
    reference_facts: dict[str, tuple[str, ...]],
    output_dir: str | Path,
) -> dict:


    raw_records = _load_jsonl(
        raw_results_path
    )

    annotation_records = _load_jsonl(
        annotations_path
    )

    raw_by_id = {
        record["question_id"]: record
        for record in raw_records
    }

    annotation_by_id = {
        record["question_id"]: record
        for record in annotation_records
    }

    question_by_id = {
        question.question_id: question
        for question in questions
    }



    missing_annotations = (
        set(raw_by_id)
        - set(annotation_by_id)
    )

    if missing_annotations:
        raise ValueError(
            "Missing annotations for: "
            + ", ".join(
                sorted(missing_annotations)
            )
        )

    missing_questions = (
        set(raw_by_id)
        - set(question_by_id)
    )

    if missing_questions:
        raise ValueError(
            "Questions not found in golden set: "
            + ", ".join(
                sorted(missing_questions)
            )
        )


    per_sample: list[dict] = []

    for question_id, raw_record in raw_by_id.items():
        question = question_by_id[
            question_id
        ]

        annotation_record = (
            annotation_by_id[
                question_id
            ]
        )

        annotation = annotation_record[
            "annotation"
        ]

        sample_metrics = (
            evaluate_generation_sample(
                question=question,
                reference_facts=(
                    reference_facts.get(
                        question_id,
                        (),
                    )
                ),
                raw_record=raw_record,
                annotation=annotation,
            )
        )

        per_sample.append(
            sample_metrics
        )



    summary = _aggregate_macro(
        per_sample=per_sample,
        question_by_id=question_by_id,
    )



    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    _save_jsonl(
        records=per_sample,
        path=(
            output_dir
            / "per_sample_metrics.jsonl"
        ),
    )

    with (
        output_dir
        / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    _save_summary_csv(
        summary=summary,
        path=(
            output_dir
            / "summary.csv"
        ),
    )

    return summary


def _aggregate_macro(
    *,
    per_sample: list[dict],
    question_by_id: dict[
        str,
        GoldenQuestion,
    ],
) -> dict:


    factual_precision = _mean_defined(
        per_sample,
        "factual_precision",
    )

    factual_recall = _mean_defined(
        per_sample,
        "factual_recall",
    )

    factual_f1 = _mean_defined(
        per_sample,
        "factual_f1",
    )


    faithfulness = _mean_defined(
        per_sample,
        "faithfulness",
    )



    citation_precision = _mean_defined(
        per_sample,
        "citation_precision",
    )

    citation_recall = _mean_defined(
        per_sample,
        "citation_recall",
    )

    citation_f1 = _mean_defined(
        per_sample,
        "citation_f1",
    )



    rouge_l_f1 = _mean_defined(
        per_sample,
        "rouge_l_f1",
    )

    bleu = _mean_defined(
        per_sample,
        "bleu",
    )


    (
        correct_abstentions,
        expected_abstentions,
        false_abstentions,
        answer_expected_cases,
    ) = _abstention_counts(
        per_sample=per_sample,
        question_by_id=question_by_id,
    )

    car = correct_abstention_rate(
        correct_abstentions=correct_abstentions,
        expected_abstentions=expected_abstentions,
    )

    far = false_abstention_rate(
        false_abstentions=false_abstentions,
        answer_expected_cases=answer_expected_cases,
    )



    if per_sample:
        output_format_valid_rate = (
            sum(
                1
                for row in per_sample
                if row["output_format_valid"]
            )
            / len(per_sample)
        )
    else:
        output_format_valid_rate = None



    latencies = [
        row["generation_latency_ms"]
        for row in per_sample
    ]

    tokens_per_second = [
        row["output_tokens_per_second"]
        for row in per_sample
    ]

    median_latency = (
        median(latencies)
        if latencies
        else None
    )

    p95_latency = (
        _percentile(
            latencies,
            0.95,
        )
        if latencies
        else None
    )

    mean_tokens_per_second = (
        fmean(tokens_per_second)
        if tokens_per_second
        else None
    )

    peak_vram_bytes = (
        max(
            row["peak_vram_bytes"]
            for row in per_sample
        )
        if per_sample
        else 0
    )



    return {
        "n_samples": len(per_sample),



        "factual_precision": factual_precision,
        "factual_recall": factual_recall,
        "factual_f1": factual_f1,

        "faithfulness": faithfulness,

        "citation_precision": citation_precision,
        "citation_recall": citation_recall,
        "citation_f1": citation_f1,

        "correct_abstention_rate": car,
        "false_abstention_rate": far,

        "rouge_l_f1": rouge_l_f1,
        "bleu": bleu,



        "output_format_valid_rate": (
            output_format_valid_rate
        ),

        "median_generation_latency_ms": (
            median_latency
        ),

        "p95_generation_latency_ms": (
            p95_latency
        ),

        "mean_output_tokens_per_second": (
            mean_tokens_per_second
        ),

        "peak_vram_bytes": (
            peak_vram_bytes
        ),



        "metric_sample_counts": {
            "factual_precision": (
                _count_defined(
                    per_sample,
                    "factual_precision",
                )
            ),
            "factual_recall": (
                _count_defined(
                    per_sample,
                    "factual_recall",
                )
            ),
            "factual_f1": (
                _count_defined(
                    per_sample,
                    "factual_f1",
                )
            ),
            "faithfulness": (
                _count_defined(
                    per_sample,
                    "faithfulness",
                )
            ),
            "citation_precision": (
                _count_defined(
                    per_sample,
                    "citation_precision",
                )
            ),
            "citation_recall": (
                _count_defined(
                    per_sample,
                    "citation_recall",
                )
            ),
            "citation_f1": (
                _count_defined(
                    per_sample,
                    "citation_f1",
                )
            ),
            "rouge_l_f1": (
                _count_defined(
                    per_sample,
                    "rouge_l_f1",
                )
            ),
            "bleu": (
                _count_defined(
                    per_sample,
                    "bleu",
                )
            ),
        },


        "abstention_counts": {
            "correct_abstentions": (
                correct_abstentions
            ),
            "expected_abstentions": (
                expected_abstentions
            ),
            "false_abstentions": (
                false_abstentions
            ),
            "answer_expected_cases": (
                answer_expected_cases
            ),
        },
    }


def _mean_defined(
    rows: list[dict],
    key: str,
) -> float | None:

    values = [
        row[key]
        for row in rows
        if row.get(key) is not None
    ]

    if not values:
        return None

    return float(
        fmean(values)
    )


def _count_defined(
    rows: list[dict],
    key: str,
) -> int:


    return sum(
        1
        for row in rows
        if row.get(key) is not None
    )


def _abstention_counts(
    *,
    per_sample: list[dict],
    question_by_id: dict[
        str,
        GoldenQuestion,
    ],
) -> tuple[int, int, int, int]:
    correct_abstentions = 0
    expected_abstentions = 0

    false_abstentions = 0
    answer_expected_cases = 0

    for row in per_sample:
        question = question_by_id[
            row["question_id"]
        ]

        context_type = row[
            "context_type"
        ]

        response_mode = row[
            "response_mode"
        ]



        if context_type == "gold":
            if question.is_answerable:
                expected = "answer"
            else:
                expected = "abstain"


        elif context_type == "retrieved":
            context_sufficiency = row[
                "context_sufficiency"
            ]

            if context_sufficiency == "full":
                expected = "answer"

            elif context_sufficiency == "none":
                expected = "abstain"

            elif context_sufficiency == "partial":

                continue

            else:
                raise ValueError(
                    "Unknown context_sufficiency: "
                    f"{context_sufficiency}"
                )

        else:
            raise ValueError(
                "Unknown context_type: "
                f"{context_type}"
            )



        if expected == "abstain":
            expected_abstentions += 1

            if response_mode == "abstain":
                correct_abstentions += 1

        else:
            answer_expected_cases += 1

            if response_mode == "abstain":
                false_abstentions += 1

    return (
        correct_abstentions,
        expected_abstentions,
        false_abstentions,
        answer_expected_cases,
    )


def _percentile(
    values: list[float],
    q: float,
) -> float:

    if not values:
        raise ValueError(
            "values must not be empty"
        )

    if not 0 <= q <= 1:
        raise ValueError(
            "q must be between 0 and 1"
        )

    ordered = sorted(values)

    if len(ordered) == 1:
        return float(
            ordered[0]
        )

    position = (
        (len(ordered) - 1)
        * q
    )

    lower = math.floor(
        position
    )

    upper = math.ceil(
        position
    )

    if lower == upper:
        return float(
            ordered[lower]
        )

    weight = (
        position - lower
    )

    return float(
        ordered[lower]
        * (1.0 - weight)
        + ordered[upper]
        * weight
    )


def _load_jsonl(
    path: str | Path,
) -> list[dict]:
    path = Path(path)

    records: list[dict] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(
                    line
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at line "
                    f"{line_number}: {path}"
                ) from exc

            records.append(
                record
            )

    return records


def _save_jsonl(
    *,
    records: list[dict],
    path: str | Path,
) -> None:
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def _save_summary_csv(
    *,
    summary: dict,
    path: str | Path,
) -> None:

    path = Path(path)

    flat_summary = {
        key: value
        for key, value in summary.items()
        if not isinstance(
            value,
            dict,
        )
    }

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                flat_summary.keys()
            ),
        )

        writer.writeheader()
        writer.writerow(
            flat_summary
        )