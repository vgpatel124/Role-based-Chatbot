import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class Settings(BaseModel):
    # Which provider answers questions (query rewriting and answer generation).
    # Groq is the default; the evaluation workflow runs the chatbot on Anthropic.
    LLM_PROVIDER: Literal["groq", "anthropic"] = Field(default="groq")
    GROQ_API_KEY: str = Field(default="")
    GROQ_MODEL: str = Field(default="openai/gpt-oss-120b")
    ANTHROPIC_API_KEY: str = Field(default="")
    ANTHROPIC_MODEL: str = Field(default="claude-haiku-5-5")
    JWT_SECRET_KEY: str = Field(default="finsolve-secret-key-change-in-production")
    JWT_ALGORITHM: str = Field(default="HS256")
    JWT_EXPIRE_MINUTES: int = Field(default=30)
    CHROMA_DB_PATH: str = Field(default="./chroma_db")
    RESOURCES_PATH: str = Field(default="./resources")
    # Origins allowed to call the API from a browser (comma-separated in the env)
    CORS_ORIGINS: list[str] = Field(default_factory=lambda: ["http://localhost:8501"])

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

    # Langfuse tracing (app/utils/tracing.py). Tracing is off until both keys are set.
    # The SDK also reads LANGFUSE_TRACING_ENVIRONMENT, LANGFUSE_RELEASE and
    # LANGFUSE_SAMPLE_RATE from the environment.
    LANGFUSE_PUBLIC_KEY: str = Field(default="")
    LANGFUSE_SECRET_KEY: str = Field(default="")
    LANGFUSE_BASE_URL: str = Field(default="https://cloud.langfuse.com")
    LANGFUSE_TRACING_ENABLED: bool = Field(default=True)
    # Send questions, answers, prompts and document text to Langfuse. Off by default:
    # this app serves HR and finance records.
    LANGFUSE_CAPTURE_CONTENT: bool = Field(default=False)


settings = Settings(
    LLM_PROVIDER=os.getenv("LLM_PROVIDER", "groq").strip().lower(),
    GROQ_API_KEY=os.getenv("GROQ_API_KEY", ""),
    GROQ_MODEL=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
    ANTHROPIC_API_KEY=os.getenv("ANTHROPIC_API_KEY", ""),
    ANTHROPIC_MODEL=os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5"),
    JWT_SECRET_KEY=os.getenv("JWT_SECRET_KEY", "finsolve-secret-key-change-in-production"),
    JWT_ALGORITHM=os.getenv("JWT_ALGORITHM", "HS256"),
    JWT_EXPIRE_MINUTES=int(os.getenv("JWT_EXPIRE_MINUTES", "30")),
    CHROMA_DB_PATH=os.getenv("CHROMA_DB_PATH", "./chroma_db"),
    RESOURCES_PATH=os.getenv("RESOURCES_PATH", "./resources"),
    CORS_ORIGINS=[o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:8501").split(",") if o.strip()],
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
    LANGFUSE_PUBLIC_KEY=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
    LANGFUSE_SECRET_KEY=os.getenv("LANGFUSE_SECRET_KEY", ""),
    LANGFUSE_BASE_URL=os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
    LANGFUSE_TRACING_ENABLED=_env_bool("LANGFUSE_TRACING_ENABLED", True),
    LANGFUSE_CAPTURE_CONTENT=_env_bool("LANGFUSE_CAPTURE_CONTENT", False),
)
