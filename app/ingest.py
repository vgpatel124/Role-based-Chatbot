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


def load_documents_by_role(data_root: Path) -> dict[str, list]:
    docs_by_collection: dict[str, list] = {collection: [] for collection in ROLE_TO_COLLECTION.values()}
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

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

            split_docs = splitter.split_documents(docs)
            docs_by_collection[collection].extend(split_docs)

    return docs_by_collection


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

    print("Ingestion completed successfully.")


if __name__ == "__main__":
    build_vector_stores()
