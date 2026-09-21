from functools import lru_cache

from langchain_community.embeddings import HuggingFaceEmbeddings

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMS = 384


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Embedding model shared by retrieval and the semantic cache (loaded once per process)."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
