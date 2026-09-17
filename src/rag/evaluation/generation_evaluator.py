from __future__ import annotations

from collections.abc import Sequence

from rag.evaluation.generation_metrics import (
    bleu_score,
    citation_precision,
    citation_recall,
    factual_precision,
    factual_recall,
    f1_score,
    faithfulness,
    rouge_l_f1,
)
from rag.evaluation.models import GoldenQuestion


def evaluate_generation_sample(
    *,
    question: GoldenQuestion,
    reference_facts: Sequence[str],
    raw_record: dict,
    annotation: dict,
) -> dict:

    claims = annotation["claims"]
    response_mode = annotation["response_mode"]


    total_claims = len(claims)

    correct_claims = sum(
        1
        for claim in claims
        if claim["correct_against_gold"]
    )

    supported_claims = sum(
        1
        for claim in claims
        if claim["supported_by_context"]
    )

    claims_with_valid_citation = sum(
        1
        for claim in claims
        if claim["citation_supported"]
    )



    covered_reference_fact_indices = set(
        annotation["covered_reference_fact_indices"]
    )

    covered_reference_facts = len(
        covered_reference_fact_indices
    )

    total_reference_facts = len(reference_facts)


    citation_annotations = annotation[
        "citation_annotations"
    ]

    selected_citations = len(
        raw_record.get(
            "used_fragment_indices",
            [],
        )
    )

    valid_citations = sum(
        1
        for citation in citation_annotations
        if citation["valid"]
    )


    if question.is_answerable:
        factual_p = factual_precision(
            correct_claims=correct_claims,
            total_claims=total_claims,
        )

        factual_r = factual_recall(
            covered_reference_facts=covered_reference_facts,
            total_reference_facts=total_reference_facts,
        )

        if (
            total_reference_facts > 0
            and total_claims == 0
        ):
            factual_f1 = 0.0
        else:
            factual_f1 = f1_score(
                precision=factual_p,
                recall=factual_r,
            )

    else:
        factual_p = None
        factual_r = None
        factual_f1 = None



    faithfulness_value = faithfulness(
        supported_claims=supported_claims,
        total_claims=total_claims,
    )



    citation_p = citation_precision(
        valid_citations=valid_citations,
        selected_citations=selected_citations,
    )

    citation_r = citation_recall(
        claims_with_valid_citation=claims_with_valid_citation,
        total_claims=total_claims,
    )


    if (
        total_claims > 0
        and selected_citations == 0
    ):
        citation_f1 = 0.0
    else:
        citation_f1 = f1_score(
            precision=citation_p,
            recall=citation_r,
        )



    if (
        question.is_answerable
        and question.reference_answer.strip()
    ):
        rouge = rouge_l_f1(
            prediction=raw_record["answer"],
            reference=question.reference_answer,
        )

        bleu = bleu_score(
            prediction=raw_record["answer"],
            reference=question.reference_answer,
        )

    else:
        rouge = None
        bleu = None



    return {
        "question_id": question.question_id,
        "configuration": raw_record["configuration"],
        "model_id": raw_record["model_id"],
        "context_type": raw_record["context_type"],
        "is_answerable": question.is_answerable,


        "total_claims": total_claims,
        "correct_claims": correct_claims,
        "supported_claims": supported_claims,

        "total_reference_facts": total_reference_facts,
        "covered_reference_facts": covered_reference_facts,

        "selected_citations": selected_citations,
        "valid_citations": valid_citations,
        "claims_with_valid_citation": (
            claims_with_valid_citation
        ),


        "factual_precision": factual_p,
        "factual_recall": factual_r,
        "factual_f1": factual_f1,


        "faithfulness": faithfulness_value,


        "citation_precision": citation_p,
        "citation_recall": citation_r,
        "citation_f1": citation_f1,


        "rouge_l_f1": rouge,
        "bleu": bleu,


        "response_mode": response_mode,
        "context_sufficiency": annotation[
            "context_sufficiency"
        ],


        "output_format_valid": raw_record.get(
            "output_format_valid",
            False,
        ),

        "generation_latency_ms": raw_record[
            "generation_latency_ms"
        ],

        "output_tokens": raw_record[
            "output_tokens"
        ],

        "output_tokens_per_second": raw_record[
            "output_tokens_per_second"
        ],

        "peak_vram_bytes": raw_record[
            "peak_vram_bytes"
        ],
    }