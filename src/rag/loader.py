from pathlib import Path
import yaml

from rag.models import Document


def load_documents(prepared_dir: str | Path) -> list[Document]:

    prepared_dir = Path(prepared_dir)

    if not prepared_dir.exists():
        raise FileNotFoundError(
            f"Prepared documents directory does not exist: {prepared_dir}"
        )

    metadata_path = prepared_dir / "documents.yaml"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Document metadata file not found: {metadata_path}"
        )

    with metadata_path.open("r", encoding="utf-8") as file:
        document_metadata = yaml.safe_load(file)

    if not isinstance(document_metadata, dict) or "documents" not in document_metadata:
        raise ValueError(
            "documents.yaml must contain a top-level 'documents' field."
        )

    document_entries = document_metadata["documents"]

    if not isinstance(document_entries, list):
        raise ValueError(
            "'documents' in documents.yaml must be a list."
        )

    documents: list[Document] = []
    seen_ids: set[str] = set()

    for entry in document_entries:
        document_id = entry.get("document_id")
        title = entry.get("title")
        source_url = entry.get("source_url")

        if not document_id:
            raise ValueError(
                "Every document must have a non-empty 'document_id'."
            )

        if document_id in seen_ids:
            raise ValueError(
                f"Duplicate document_id in documents.yaml: {document_id}"
            )

        if not title:
            raise ValueError(
                f"Document '{document_id}' has no title."
            )

        markdown_path = prepared_dir / f"{document_id}.md"

        if not markdown_path.exists():
            raise FileNotFoundError(
                f"Markdown file for document '{document_id}' "
                f"not found: {markdown_path}"
            )

        text = markdown_path.read_text(encoding="utf-8").strip()

        if not text:
            raise ValueError(
                f"Markdown document is empty: {markdown_path}"
            )

        documents.append(
            Document(
                document_id=document_id,
                title=title,
                text=text,
                source_url=source_url,
                source_path=markdown_path,
            )
        )

        seen_ids.add(document_id)

    return documents