import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class Settings(BaseModel):
    GROQ_API_KEY: str = Field(default="")
    GROQ_MODEL: str = Field(default="openai/gpt-oss-120b")
    JWT_SECRET_KEY: str = Field(default="finsolve-secret-key-change-in-production")
    JWT_ALGORITHM: str = Field(default="HS256")
    JWT_EXPIRE_MINUTES: int = Field(default=30)
    CHROMA_DB_PATH: str = Field(default="./chroma_db")
    RESOURCES_PATH: str = Field(default="./resources")

    # Redis Stack Semantic Cache
    REDIS_URL: str = Field(default="redis://localhost:6379")
    CACHE_TTL: int = Field(default=3600)
    CACHE_SIMILARITY_THRESHOLD: float = Field(default=0.92)
    CACHE_ENABLED: bool = Field(default=True)

    # Retrieval
    DENSE_TOP_K: int = Field(default=15)
    BM25_TOP_K: int = Field(default=15)
    RRF_K: int = Field(default=60)
    RRF_CANDIDATE_COUNT: int = Field(default=15)

    # Reranker
    RERANKER_MODEL: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    RERANKER_CANDIDATE_COUNT: int = Field(default=15)
    RERANKER_FINAL_TOP_K: int = Field(default=6)
    RERANKER_SCORE_THRESHOLD: float = Field(default=0.1)

    # Conversation
    CONVERSATION_HISTORY_LENGTH: int = Field(default=6)
    QUERY_REWRITING_ENABLED: bool = Field(default=True)

    # Cache versioning — increment to invalidate all cached responses
    RETRIEVAL_VERSION: str = Field(default="v2")


settings = Settings(
    GROQ_API_KEY=os.getenv("GROQ_API_KEY", ""),
    GROQ_MODEL=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
    JWT_SECRET_KEY=os.getenv("JWT_SECRET_KEY", "finsolve-secret-key-change-in-production"),
    JWT_ALGORITHM=os.getenv("JWT_ALGORITHM", "HS256"),
    JWT_EXPIRE_MINUTES=int(os.getenv("JWT_EXPIRE_MINUTES", "30")),
    CHROMA_DB_PATH=os.getenv("CHROMA_DB_PATH", "./chroma_db"),
    RESOURCES_PATH=os.getenv("RESOURCES_PATH", "./resources"),
    REDIS_URL=os.getenv("REDIS_URL", "redis://localhost:6379"),
    CACHE_TTL=int(os.getenv("CACHE_TTL", "3600")),
    CACHE_SIMILARITY_THRESHOLD=float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.92")),
    CACHE_ENABLED=_env_bool("CACHE_ENABLED", True),
    DENSE_TOP_K=int(os.getenv("DENSE_TOP_K", "15")),
    BM25_TOP_K=int(os.getenv("BM25_TOP_K", "15")),
    RRF_K=int(os.getenv("RRF_K", "60")),
    RRF_CANDIDATE_COUNT=int(os.getenv("RRF_CANDIDATE_COUNT", "15")),
    RERANKER_MODEL=os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
    RERANKER_CANDIDATE_COUNT=int(os.getenv("RERANKER_CANDIDATE_COUNT", "15")),
    RERANKER_FINAL_TOP_K=int(os.getenv("RERANKER_FINAL_TOP_K", "6")),
    RERANKER_SCORE_THRESHOLD=float(os.getenv("RERANKER_SCORE_THRESHOLD", "0.1")),
    CONVERSATION_HISTORY_LENGTH=int(os.getenv("CONVERSATION_HISTORY_LENGTH", "6")),
    QUERY_REWRITING_ENABLED=_env_bool("QUERY_REWRITING_ENABLED", True),
    RETRIEVAL_VERSION=os.getenv("RETRIEVAL_VERSION", "v2"),
)
