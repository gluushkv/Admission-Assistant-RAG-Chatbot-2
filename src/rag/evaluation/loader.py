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

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in golden set: {path}"
            ) from exc

    if not isinstance(data, list):
        raise ValueError(
            "Golden set JSON must contain a top-level list."
        )

    if not data:
        raise ValueError(
            f"Golden set is empty: {path}"
        )

    questions = [
        _parse_golden_question(
            item,
            index=index,
        )
        for index, item in enumerate(data)
    ]

    return questions


def _parse_golden_question(
    data: dict,
    *,
    index: int,
) -> GoldenQuestion:
    if not isinstance(data, dict):
        raise ValueError(
            f"Golden question at index {index} "
            "must be a JSON object."
        )

    try:
        evidence = [
            _parse_evidence(
                item,
                question_index=index,
                evidence_index=evidence_index,
            )
            for evidence_index, item
            in enumerate(data.get("evidence", []))
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

    except KeyError as exc:
        raise ValueError(
            f"Golden question at index {index} "
            f"is missing required field: {exc.args[0]}"
        ) from exc


def _parse_evidence(
    data: dict,
    *,
    question_index: int,
    evidence_index: int,
) -> Evidence:
    if not isinstance(data, dict):
        raise ValueError(
            f"Evidence {evidence_index} in question "
            f"{question_index} must be a JSON object."
        )

    try:
        return Evidence(
            evidence_id=data["evidence_id"],
            document_id=data["document_id"],
            text=data["text"],
            start_offset=data["start_offset"],
            end_offset=data["end_offset"],
        )

    except KeyError as exc:
        raise ValueError(
            f"Evidence {evidence_index} in question "
            f"{question_index} is missing required field: "
            f"{exc.args[0]}"
        ) from exc


def load_reference_facts(
    path: str | Path,
) -> dict[str, tuple[str, ...]]:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            "Reference facts file "
            f"not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(
            file
        )

    if not isinstance(
        data,
        list,
    ):
        raise ValueError(
            "Reference facts JSON "
            "must contain a list"
        )

    result: dict[
        str,
        tuple[str, ...],
    ] = {}

    for item in data:
        if not isinstance(
            item,
            dict,
        ):
            raise ValueError(
                "Reference facts item "
                "must be an object"
            )

        question_id = item.get(
            "question_id"
        )

        facts = item.get(
            "reference_facts"
        )

        if not isinstance(
            question_id,
            str,
        ):
            raise ValueError(
                "Invalid question_id"
            )

        if not isinstance(
            facts,
            list,
        ):
            raise ValueError(
                "reference_facts "
                "must be a list"
            )

        parsed_facts: list[str] = []

        for fact in facts:
            if not isinstance(
                fact,
                str,
            ):
                raise ValueError(
                    "Reference fact "
                    "must be a string"
                )

            parsed_facts.append(
                fact.strip()
            )

        result[
            question_id
        ] = tuple(
            parsed_facts
        )

    return result