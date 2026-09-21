import json
import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from app.config import settings
except ModuleNotFoundError:
    # Supports direct execution: `python app/ingest.py`
    from config import settings

ROLE_TO_COLLECTION = {
    "finance": "finance",
    "marketing": "marketing",
    # Chroma requires collection names with 3+ characters.
    "hr": "hr_docs",
    "engineering": "engineering",
    "employee": "general",
}

FOLDER_TO_ROLE = {
    "finance": "finance",
    "marketing": "marketing",
    "hr": "hr",
    "engineering": "engineering",
    "general": "employee",
}


def resolve_data_root() -> Path:
    resources_path = Path(settings.RESOURCES_PATH)
    data_path = resources_path / "data"
    return data_path if data_path.exists() else resources_path


# --- Contextual chunk headers ------------------------------------------------
# Every chunk is prefixed with "<document title> › <section>" before it is
# embedded (Chroma) and persisted (BM25), e.g.
#     "Comprehensive Marketing Report - Q2 2024 › Campaign Highlights"
# Without it, repeated sections such as "## Campaign Highlights" (present in
# every quarterly marketing report) carry no quarter or report identity and
# cannot be told apart by retrieval. The header goes FIRST because the
# embedding model truncates long inputs (~256 tokens for all-MiniLM-L6-v2).

CONTEXT_HEADER_SEPARATOR = " › "

_ATX_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_SETEXT_UNDERLINE = re.compile(r"^\s*-{3,}\s*$")
_LIST_ITEM = re.compile(r"^\s*([-*+]|\d+\.)\s")


def title_from_filename(file_name: str) -> str:
    """hr_data.csv -> "HR Data"; marketing_report_q2_2024.md -> "Marketing Report Q2 2024"."""
    words = re.split(r"[_\-\s]+", Path(file_name).stem)
    return " ".join(w.upper() if re.fullmatch(r"[a-z]{1,2}|[a-z]\d+", w.lower()) else w.capitalize() for w in words if w)


def markdown_outline(text: str) -> tuple[str | None, list[tuple[int, str]]]:
    """Return (H1 title, [(char offset, section name), ...]) for a markdown text.

    Sections are level-2 headings: ATX ("## Campaign Highlights") or setext
    ("Executive Summary:" underlined with dashes, used by the annual reports).
    """
    title: str | None = None
    sections: list[tuple[int, str]] = []
    lines = text.splitlines(keepends=True)
    offset = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        atx = _ATX_HEADING.match(stripped)
        if atx:
            level, name = len(atx.group(1)), atx.group(2).strip()
            if level == 1 and title is None:
                title = name
            elif level == 2:
                sections.append((offset, name))
        elif (
            stripped
            and i + 1 < len(lines)
            and _SETEXT_UNDERLINE.match(lines[i + 1])
            and not _LIST_ITEM.match(line)
        ):
            sections.append((offset, stripped.rstrip(":").strip()))
        offset += len(line)
    return title, sections


def dominant_section(sections: list[tuple[int, str]], start: int, end: int) -> str | None:
    """The section covering the most characters of text[start:end].

    A chunk that begins with overlap from the previous section is labelled
    with the section most of it actually belongs to.
    """
    best_name, best_overlap = None, 0
    for i, (sec_start, name) in enumerate(sections):
        sec_end = sections[i + 1][0] if i + 1 < len(sections) else float("inf")
        overlap = min(end, sec_end) - max(start, sec_start)
        if overlap > best_overlap:
            best_name, best_overlap = name, overlap
    return best_name


def build_context_header(source_text: str, chunk_text: str, start_index: int, file_name: str, suffix: str) -> str:
    if suffix == ".csv":
        # CSVLoader: one document per row, first line is "<first column>: <value>".
        first_field = chunk_text.strip().splitlines()[0].strip() if chunk_text.strip() else ""
        parts = [title_from_filename(file_name), first_field]
    else:
        title, sections = markdown_outline(source_text)
        section = None
        if start_index >= 0:
            section = dominant_section(sections, start_index, start_index + len(chunk_text))
        parts = [title or title_from_filename(file_name), section]
    return CONTEXT_HEADER_SEPARATOR.join(p for p in parts if p)


def add_context_headers(source_doc, chunks: list, file_name: str, suffix: str) -> list:
    """Prefix each chunk's page_content with its header; metadata is kept and
    extended with "context_header"."""
    for chunk in chunks:
        header = build_context_header(
            source_doc.page_content,
            chunk.page_content,
            chunk.metadata.get("start_index", -1),
            file_name,
            suffix,
        )
        chunk.metadata["context_header"] = header
        chunk.page_content = f"{header}\n\n{chunk.page_content}"
    return chunks


def load_documents_by_role(data_root: Path) -> dict[str, list]:
    docs_by_collection: dict[str, list] = {collection: [] for collection in ROLE_TO_COLLECTION.values()}
    # add_start_index records where each chunk begins in its source document,
    # which is how the chunk's section is located. Chunk boundaries are
    # unchanged (same size/overlap); the header is added after splitting.
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)

    for folder_name, role in FOLDER_TO_ROLE.items():
        folder = data_root / folder_name
        if not folder.exists():
            continue

        collection = ROLE_TO_COLLECTION[role]
        for file_path in folder.rglob("*"):
            if not file_path.is_file():
                continue

            suffix = file_path.suffix.lower()
            if suffix == ".pdf":
                loader = PyPDFLoader(str(file_path))
            elif suffix == ".csv":
                loader = CSVLoader(str(file_path))
            elif suffix in [".txt", ".md"]:
                loader = TextLoader(str(file_path), encoding="utf-8")
            else:
                continue

            docs = loader.load()
            for doc in docs:
                doc.metadata["role"] = role
                doc.metadata["source"] = file_path.name

            for doc in docs:
                chunks = splitter.split_documents([doc])
                docs_by_collection[collection].extend(add_context_headers(doc, chunks, file_path.name, suffix))

    return docs_by_collection


BM25_METADATA_FIELDS = ("source", "role")


def save_bm25_docs(docs_by_collection: dict[str, list], persist_directory: str) -> None:
    """Persist chunk text per collection so BM25 can be rebuilt at query time.

    Only "source" and "role" are kept from metadata; CSV row fields or any
    other loader metadata are dropped.
    """
    bm25_dir = Path(persist_directory) / "bm25_docs"
    bm25_dir.mkdir(parents=True, exist_ok=True)

    for collection_name, documents in docs_by_collection.items():
        if not documents:
            continue

        records = [
            {
                "page_content": doc.page_content,
                "metadata": {key: doc.metadata.get(key) for key in BM25_METADATA_FIELDS},
            }
            for doc in documents
        ]
        output_path = bm25_dir / f"{collection_name}.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False)
        print(f"BM25 docs saved: {collection_name} → {len(records)} chunks")


def build_vector_stores() -> None:
    data_root = resolve_data_root()
    if not data_root.exists():
        raise FileNotFoundError(f"Resources path not found: {data_root}")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    docs_by_collection = load_documents_by_role(data_root)
    persist_directory = settings.CHROMA_DB_PATH

    for collection_name, documents in docs_by_collection.items():
        if not documents:
            continue

        Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=persist_directory,
        )
        print(f"Ingested {len(documents)} chunks into collection: {collection_name}")

    save_bm25_docs(docs_by_collection, persist_directory)

    print("Ingestion completed successfully.")


if __name__ == "__main__":
    build_vector_stores()
