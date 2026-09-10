import argparse
import json
from pathlib import Path

CONTEXT_CHARS = 0  # сколько символов до/после evidence показывать


def load_documents(prepared_dir: Path) -> dict[str, str]:
    documents = {}
    for path in prepared_dir.glob("*.md"):
        documents[path.stem] = path.read_text(encoding="utf-8")
    return documents


def format_snippet(doc_text: str, start: int, end: int) -> str:
    before = doc_text[max(0, start - CONTEXT_CHARS):start].replace("\n", "⏎")
    evidence = doc_text[start:end].replace("\n", "⏎")
    after = doc_text[end:end + CONTEXT_CHARS].replace("\n", "⏎")
    return f"...{before}【{evidence}】{after}..."


def inspect(golden_path: Path, prepared_dir: Path) -> None:
    documents = load_documents(prepared_dir)

    with open(golden_path, encoding="utf-8") as f:
        questions = json.load(f)

    shown = 0
    unresolved = 0

    for q in questions:
        if not q.get("evidence"):
            continue

        print(f"\n{'='*80}")
        print(f"[{q['question_id']}] ({q.get('formulation_type')}, relation={q.get('evidence_relation')})")
        print(f"Вопрос: {q['question']}")
        print(f"Ожидаемый ответ: {q.get('reference_answer')}")
        print("-" * 80)

        for e in q["evidence"]:
            start, end = e.get("start_offset"), e.get("end_offset")
            doc_id = e["document_id"]

            if start is None or end is None:
                print(f"  [НЕ РАЗРЕШЕНО] doc={doc_id}, evidence_id={e['evidence_id']}")
                unresolved += 1
                continue

            doc_text = documents.get(doc_id)
            if doc_text is None:
                print(f"  [ДОКУМЕНТ НЕ НАЙДЕН] doc={doc_id}")
                continue

            snippet = format_snippet(doc_text, start, end)
            print(f"  doc={doc_id}  [{start}:{end}]")
            print(f"  {snippet}")
            shown += 1

    print(f"\n{'='*80}")
    print(f"Показано evidence: {shown} | Нерезолвленных: {unresolved}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    inspect(
        golden_path=Path("data/golden_set_with_offsets.json"),
        prepared_dir=Path("data/prepared/"),
    )