from __future__ import annotations

import re

from nltk.translate.bleu_score import (
    SmoothingFunction,
    sentence_bleu,
)


def factual_precision(
    *,
    correct_claims: int,
    total_claims: int,
) -> float | None:
    if total_claims == 0:
        return None

    return (
        correct_claims
        / total_claims
    )


def factual_recall(
    *,
    covered_reference_facts: int,
    total_reference_facts: int,
) -> float | None:
    if total_reference_facts == 0:
        return None

    return (
        covered_reference_facts
        / total_reference_facts
    )


def f1_score(
    *,
    precision: float | None,
    recall: float | None,
) -> float | None:
    if (
        precision is None
        or recall is None
    ):
        return None

    if precision + recall == 0:
        return 0.0

    return (
        2.0
        * precision
        * recall
        / (
            precision
            + recall
        )
    )


def faithfulness(
    *,
    supported_claims: int,
    total_claims: int,
) -> float | None:
    if total_claims == 0:
        return None

    return (
        supported_claims
        / total_claims
    )


def citation_precision(
    *,
    valid_citations: int,
    selected_citations: int,
) -> float | None:
    if selected_citations == 0:
        return None

    return (
        valid_citations
        / selected_citations
    )


def citation_recall(
    *,
    claims_with_valid_citation: int,
    total_claims: int,
) -> float | None:
    if total_claims == 0:
        return None

    return (
        claims_with_valid_citation
        / total_claims
    )


def correct_abstention_rate(
    *,
    correct_abstentions: int,
    expected_abstentions: int,
) -> float | None:
    if expected_abstentions == 0:
        return None

    return (
        correct_abstentions
        / expected_abstentions
    )


def false_abstention_rate(
    *,
    false_abstentions: int,
    answer_expected_cases: int,
) -> float | None:
    if answer_expected_cases == 0:
        return None

    return (
        false_abstentions
        / answer_expected_cases
    )


def rouge_l_f1(
    *,
    prediction: str,
    reference: str,
) -> float | None:
    prediction_tokens = _tokenize(
        prediction
    )

    reference_tokens = _tokenize(
        reference
    )

    if not prediction_tokens:
        return 0.0

    if not reference_tokens:
        return None

    lcs = _lcs_length(
        prediction_tokens,
        reference_tokens,
    )

    precision = (
        lcs
        / len(prediction_tokens)
    )

    recall = (
        lcs
        / len(reference_tokens)
    )

    if precision + recall == 0:
        return 0.0

    return (
        2.0
        * precision
        * recall
        / (
            precision
            + recall
        )
    )


def bleu_score(
    *,
    prediction: str,
    reference: str,
) -> float | None:
    prediction_tokens = _tokenize(
        prediction
    )

    reference_tokens = _tokenize(
        reference
    )

    if not reference_tokens:
        return None

    if not prediction_tokens:
        return 0.0

    smoothing = (
        SmoothingFunction()
        .method1
    )

    return float(
        sentence_bleu(
            [reference_tokens],
            prediction_tokens,
            weights=(
                0.25,
                0.25,
                0.25,
                0.25,
            ),
            smoothing_function=(
                smoothing
            ),
        )
    )


def _tokenize(
    text: str,
) -> list[str]:
    return re.findall(
        r"\w+",
        text.lower(),
        flags=re.UNICODE,
    )


def _lcs_length(
    left: list[str],
    right: list[str],
) -> int:

    previous = [
        0
    ] * (
        len(right) + 1
    )

    for left_token in left:
        current = [
            0
        ] * (
            len(right) + 1
        )

        for index, right_token in enumerate(
            right,
            start=1,
        ):
            if left_token == right_token:
                current[index] = (
                    previous[
                        index - 1
                    ]
                    + 1
                )
            else:
                current[index] = max(
                    current[
                        index - 1
                    ],
                    previous[index],
                )

        previous = current

    return previous[-1]