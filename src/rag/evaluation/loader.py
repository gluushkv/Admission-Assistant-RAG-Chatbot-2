from __future__ import annotations

import json
from pathlib import Path

from rag.evaluation.models import (
    Evidence,
    GoldenQuestion,
)


def load_golden_questions(
    path: str | Path,
) -> list[GoldenQuestion]:

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Golden set file not found: {path}"
        )

    questions: list[GoldenQuestion] = []

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
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at "
                    f"{path}:{line_number}"
                ) from exc

            questions.append(
                _parse_golden_question(
                    data,
                    line_number=line_number,
                )
            )

    if not questions:
        raise ValueError(
            f"Golden set is empty: {path}"
        )

    return questions


def _parse_golden_question(
    data: dict,
    *,
    line_number: int,
) -> GoldenQuestion:
    try:
        evidence = [
            Evidence(
                evidence_id=item["evidence_id"],
                document_id=item["document_id"],
                text=item["text"],
                start_offset=item["start_offset"],
                end_offset=item["end_offset"],
            )
            for item in data.get("evidence", [])
        ]

        return GoldenQuestion(
            question_id=data["question_id"],
            formulation_type=data["formulation_type"],
            question=data["question"],
            is_answerable=data["is_answerable"],
            reference_answer=data.get(
                "reference_answer"
            ),
            evidence_relation=data.get(
                "evidence_relation"
            ),
            evidence=evidence,
        )

    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Invalid golden question "
            f"at line {line_number}."
        ) from exc