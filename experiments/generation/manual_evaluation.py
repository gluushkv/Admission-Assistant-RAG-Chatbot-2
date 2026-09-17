from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from rag.evaluation.models import (
    GoldenQuestion,
)


def create_annotation_template(
    *,
    raw_results_path: str | Path,
    questions: Sequence[
        GoldenQuestion
    ],
    reference_facts: Mapping[
        str,
        Sequence[str],
    ],
    output_path: str | Path,
) -> None:
    raw_records = _load_jsonl(
        raw_results_path
    )

    question_by_id = {
        question.question_id: question
        for question in questions
    }

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for raw in raw_records:
            question_id = raw[
                "question_id"
            ]

            question = question_by_id[
                question_id
            ]

            facts = list(
                reference_facts.get(
                    question_id,
                    (),
                )
            )

            context_type = raw[
                "context_type"
            ]

            if context_type == "gold":
                context_sufficiency = (
                    "full"
                    if question.is_answerable
                    else "none"
                )
            else:
                context_sufficiency = None

            annotation = {
                "response_mode": None,

                "context_sufficiency": (
                    context_sufficiency
                ),

                "claims": [
                ],

                "covered_reference_fact_indices": [
                ],

                "citation_annotations": [
                    {
                        "fragment_index": (
                            fragment_index
                        ),
                        "valid": None,
                    }
                    for fragment_index
                    in raw.get(
                        "used_fragment_indices",
                        [],
                    )
                ],
            }

            record = {
                "question_id": question_id,
                "configuration": raw[
                    "configuration"
                ],
                "model_id": raw[
                    "model_id"
                ],
                "context_type": (
                    context_type
                ),

                "question": (
                    question.question
                ),

                "is_answerable": (
                    question.is_answerable
                ),

                "generated_answer": (
                    raw["answer"]
                ),

                "gold_evidence": [
                    {
                        "evidence_id": (
                            evidence.evidence_id
                        ),
                        "document_id": (
                            evidence.document_id
                        ),
                        "text": (
                            evidence.text
                        ),
                    }
                    for evidence
                    in question.evidence
                ],

                "reference_facts": (
                    facts
                ),

                "actual_context": (
                    raw["context"]
                ),

                "used_fragment_indices": (
                    raw.get(
                        "used_fragment_indices",
                        [],
                    )
                ),

                "annotation": annotation,
            }

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def validate_annotations(
    path: str | Path,
) -> list[dict]:
    records = _load_jsonl(
        path
    )

    errors: list[str] = []

    for line_number, record in enumerate(
        records,
        start=1,
    ):
        annotation = record.get(
            "annotation"
        )

        if not isinstance(
            annotation,
            dict,
        ):
            errors.append(
                f"Line {line_number}: "
                "annotation is missing"
            )
            continue

        response_mode = annotation.get(
            "response_mode"
        )

        if response_mode not in {
            "answer",
            "abstain",
        }:
            errors.append(
                f"Line {line_number}: "
                "invalid response_mode"
            )

        context_sufficiency = (
            annotation.get(
                "context_sufficiency"
            )
        )

        if context_sufficiency not in {
            "full",
            "partial",
            "none",
        }:
            errors.append(
                f"Line {line_number}: "
                "invalid context_sufficiency"
            )

        claims = annotation.get(
            "claims"
        )

        if not isinstance(
            claims,
            list,
        ):
            errors.append(
                f"Line {line_number}: "
                "claims must be a list"
            )
            continue

        for claim_index, claim in enumerate(
            claims
        ):
            if not isinstance(
                claim,
                dict,
            ):
                errors.append(
                    f"Line {line_number}, "
                    f"claim {claim_index}: "
                    "must be an object"
                )
                continue

            if not isinstance(
                claim.get("text"),
                str,
            ):
                errors.append(
                    f"Line {line_number}, "
                    f"claim {claim_index}: "
                    "text is missing"
                )

            for field in (
                "correct_against_gold",
                "supported_by_context",
                "citation_supported",
            ):
                if not isinstance(
                    claim.get(field),
                    bool,
                ):
                    errors.append(
                        f"Line {line_number}, "
                        f"claim {claim_index}: "
                        f"{field} must be bool"
                    )

        reference_facts = record.get(
            "reference_facts",
            [],
        )

        covered = annotation.get(
            "covered_reference_fact_indices"
        )

        if not isinstance(
            covered,
            list,
        ):
            errors.append(
                f"Line {line_number}: "
                "covered_reference_fact_indices "
                "must be a list"
            )
        else:
            for index in covered:
                if (
                    not isinstance(index, int)
                    or isinstance(index, bool)
                    or index < 0
                    or index >= len(
                        reference_facts
                    )
                ):
                    errors.append(
                        f"Line {line_number}: "
                        "invalid reference fact "
                        f"index {index}"
                    )

        citations = annotation.get(
            "citation_annotations"
        )

        if not isinstance(
            citations,
            list,
        ):
            errors.append(
                f"Line {line_number}: "
                "citation_annotations "
                "must be a list"
            )
        else:
            for citation in citations:
                if not isinstance(
                    citation,
                    dict,
                ):
                    errors.append(
                        f"Line {line_number}: "
                        "invalid citation annotation"
                    )
                    continue

                if not isinstance(
                    citation.get(
                        "fragment_index"
                    ),
                    int,
                ):
                    errors.append(
                        f"Line {line_number}: "
                        "invalid fragment_index"
                    )

                if not isinstance(
                    citation.get(
                        "valid"
                    ),
                    bool,
                ):
                    errors.append(
                        f"Line {line_number}: "
                        "citation valid "
                        "must be bool"
                    )

    if errors:
        raise ValueError(
            "Annotation validation failed:\n"
            + "\n".join(
                errors
            )
        )

    return records


def _load_jsonl(
    path: str | Path,
) -> list[dict]:
    path = Path(
        path
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
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(
                    line
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at "
                    f"line {line_number}"
                ) from exc

            records.append(
                record
            )

    return records