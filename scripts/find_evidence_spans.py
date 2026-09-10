import json
from pathlib import Path


def load_documents(prepared_dir: Path) -> dict[str, str]:
    documents = {}
    for path in prepared_dir.glob("*.md"):
        documents[path.stem] = path.read_text(encoding="utf-8")
    return documents


def resolve_offsets(golden_path: Path, prepared_dir: Path, output_path: Path) -> None:
    documents = load_documents(prepared_dir)

    with open(golden_path, encoding="utf-8") as f:
        questions = json.load(f)

    errors = []
    resolved = 0
    total = 0

    for q in questions:
        for e in q.get("evidence", []):
            total += 1
            doc_id = e["document_id"]
            text = e.get("text")

            if not text:
                errors.append(
                    f"{q['question_id']} / {e['evidence_id']}: text пустой или отсутствует"
                )
                continue

            doc_text = documents.get(doc_id)
            if doc_text is None:
                errors.append(
                    f"{q['question_id']} / {e['evidence_id']}: документ '{doc_id}' "
                    f"не найден в {prepared_dir}. Проверьте имя файла .md "
                )
                continue

            count = doc_text.count(text)

            if count == 0:
                errors.append(
                    f"{q['question_id']} / {e['evidence_id']}: текст НЕ найден дословно "
                    f"Сверьте evidence.text с исходным .md."
                )
                continue

            if count > 1:
                errors.append(
                    f"{q['question_id']} / {e['evidence_id']}: текст встречается {count} раз(а) "
                    f"в документе '{doc_id}' — неоднозначно."
                )
                continue

            start = doc_text.index(text)
            end = start + len(text)
            e["start_offset"] = start
            e["end_offset"] = end
            resolved += 1

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    print(f"Всего evidence: {total}")
    print(f"Успешно разрешено: {resolved}")
    print(f"Ошибок: {len(errors)}\n")

    if errors:
        print("--- Требуют ручного исправления ---")
        for err in errors:
            print(f"  - {err}")
    else:
        print("Все offsets успешно вычислены.")


if __name__ == "__main__":
    resolve_offsets(
        golden_path=Path("data/golden_set.json"),
        prepared_dir=Path("data/prepared/"),
        output_path=Path("data/golden_set_with_offsets.json"),
    )