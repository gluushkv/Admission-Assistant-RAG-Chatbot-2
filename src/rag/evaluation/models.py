from dataclasses import dataclass

@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    document_id: str
    start_offset: int
    end_offset: int
    text: str | None = None

@dataclass(frozen=True)
class GoldenQuestion:
    question_id: str
    question: str
    formulation_type: str
    is_answerable: bool
    reference_answer: str
    evidence_relation: str | None
    evidence: tuple[Evidence, ...]