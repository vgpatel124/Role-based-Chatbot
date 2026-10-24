"""Shared fixtures. Every external dependency is faked:

* Groq LLM            → mock_llm (MagicMock; records every invoke)
* ChromaDB            → mock_chromadb (in-memory corpus; records collections touched)
* BM25 JSON on disk   → recorded fake inside mock_chromadb
* Redis Stack         → mock_redis (MagicMock client + SearchIndex)
* Cross-encoder model → FakeCrossEncoder (lexical-overlap scores, no download)
* Embedding model     → FakeEmbeddings (deterministic hash vectors, no download)

The environment and the two model classes are patched at the top of this
file, BEFORE anything from `app` is imported, because several app modules do
work at import time (settings read os.environ, the reranker loads its model,
the cache connects to Redis, main.py warms the embedding model).
"""

import hashlib
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# load_dotenv() never overrides variables that are already set, so these win
# over the developer's .env.
os.environ["CACHE_ENABLED"] = "false"  # no Redis connection at import
os.environ["REDIS_URL"] = "redis://127.0.0.1:1"
os.environ["GROQ_API_KEY"] = "test-key"
os.environ["JWT_SECRET_KEY"] = "test-secret-key"
os.environ["QUERY_REWRITING_ENABLED"] = "true"
# A developer's .env may hold Langfuse keys; empty keys keep tracing off, so tests never
# send traces (tracing tests switch it on with an in-memory exporter).
# The provider is Groq unless a test says otherwise, whatever a developer's .env contains.
os.environ["LLM_PROVIDER"] = "groq"
os.environ["ANTHROPIC_API_KEY"] = ""
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""
os.environ["RETRIEVAL_VERSION"] = "v1"
# Never read the developer's real chroma_db/ or bm25_docs/.
os.environ["CHROMA_DB_PATH"] = str(ROOT / "tests" / "_nonexistent_chroma_db")

_WORD = re.compile(r"\w+")
_FAKE_STOPWORDS = {"what", "which", "does", "with", "that", "this", "from", "have", "were", "about", "many"}


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) >= 4 and w not in _FAKE_STOPWORDS}


class FakeCrossEncoder:
    """Stand-in for sentence_transformers.CrossEncoder.

    Score = fraction of the query's content words present in the passage, so
    on-topic chunks score high and off-topic (e.g. other-department) chunks 0.
    """

    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs

    def predict(self, pairs):
        scores = []
        for query, passage in pairs:
            q = _content_words(query)
            scores.append(len(q & _content_words(passage)) / len(q) if q else 0.0)
        return np.array(scores, dtype=np.float32)


class FakeEmbeddings:
    """Deterministic 384-d unit vectors derived from the text hash."""

    def __init__(self, *args, **kwargs):
        pass

    def embed_query(self, text: str) -> list[float]:
        seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        vec = np.random.default_rng(seed).standard_normal(384)
        return (vec / np.linalg.norm(vec)).astype(np.float32).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]


import langchain_community.embeddings  # noqa: E402
import sentence_transformers  # noqa: E402

sentence_transformers.CrossEncoder = FakeCrossEncoder
langchain_community.embeddings.HuggingFaceEmbeddings = FakeEmbeddings

from jose import jwt  # noqa: E402
from langchain_core.documents import Document  # noqa: E402

import app.services.cache as cache_module  # noqa: E402
from app import rag  # noqa: E402
from app.auth import create_access_token  # noqa: E402
from app.config import settings  # noqa: E402
from app.services.session_store import session_store  # noqa: E402

ALL_COLLECTIONS = ["finance", "marketing", "hr_docs", "engineering", "general"]

USER_FOR_ROLE = {
    "finance": "alice",
    "marketing": "bob",
    "hr": "carol",
    "engineering": "dave",
    "c_level": "eve",
    "employee": "frank",
}
TEST_PASSWORD = "password123"


def _doc(text: str, source: str, role: str) -> Document:
    return Document(page_content=text, metadata={"source": source, "role": role})


CORPUS: dict[str, list[Document]] = {
    "finance": [
        _doc("Q3 2024 gross margin was 62% and revenue was $2.4 billion.", "quarterly_financial_report.md", "finance"),
        _doc("Vendor services expenses totalled $30M in 2024.", "financial_summary.md", "finance"),
    ],
    "hr_docs": [
        _doc("employee_id: FINEMP1005\nfull_name: Shaurya Joshi\nleave_balance: 4", "hr_data.csv", "hr"),
        _doc("employee_id: FINEMP1010\nfull_name: Vihaan Chowdhury\nleave_balance: 10", "hr_data.csv", "hr"),
    ],
    "marketing": [
        _doc("Q2 2024 influencer partnerships drove 40,000 customer sign-ups.", "marketing_report_q2_2024.md", "marketing"),
    ],
    "engineering": [
        _doc("Microservices are deployed and scaled on Kubernetes.", "engineering_master_doc.md", "engineering"),
    ],
    "general": [
        _doc("Employees get 12 days of sick leave per year.", "employee_handbook.md", "employee"),
    ],
}


class FakeVectorStore:
    def __init__(self, name: str, recorder: SimpleNamespace):
        self._name = name
        self._recorder = recorder

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        self._recorder.dense_calls.append((self._name, query))
        return [Document(page_content=d.page_content, metadata=dict(d.metadata)) for d in CORPUS.get(self._name, [])][:k]


@pytest.fixture(autouse=True)
def _isolate_state():
    """Fresh conversation history for every test."""
    session_store._store.clear()
    yield
    session_store._store.clear()


@pytest.fixture
def mock_llm(monkeypatch):
    llm = MagicMock(name="groq_llm")
    llm.invoke.return_value = SimpleNamespace(content="Mocked answer [source.md]")
    monkeypatch.setattr(rag, "get_llm", lambda: llm)
    return llm


@pytest.fixture
def mock_chromadb(monkeypatch):
    """In-memory Chroma + BM25. Records every collection that is searched."""
    recorder = SimpleNamespace(dense_calls=[], bm25_calls=[])

    def fake_bm25(question: str, collection_name: str, top_k: int) -> list[Document]:
        recorder.bm25_calls.append((collection_name, question))
        return [Document(page_content=d.page_content, metadata=dict(d.metadata)) for d in CORPUS.get(collection_name, [])][:top_k]

    monkeypatch.setattr(rag, "_get_vectorstore", lambda name, embeddings: FakeVectorStore(name, recorder))
    monkeypatch.setattr(rag.bm25_retriever, "retrieve", fake_bm25)
    recorder.touched = lambda: {c for c, _ in recorder.dense_calls} | {c for c, _ in recorder.bm25_calls}
    return recorder


@pytest.fixture
def mock_redis(monkeypatch):
    """Redis client + redisvl SearchIndex mocks for RBACSemanticCache."""
    client = MagicMock(name="redis_client")
    index = MagicMock(name="search_index")
    index.exists.return_value = True
    index.schema.index.prefix = "rbac_cache_v1"
    index.load.return_value = ["rbac_cache_v1:01TESTKEY"]
    index.query.return_value = []
    monkeypatch.setattr(cache_module.redis, "from_url", lambda *args, **kwargs: client)
    monkeypatch.setattr(cache_module, "SearchIndex", lambda schema, redis_client=None: index)
    monkeypatch.setattr(settings, "CACHE_ENABLED", True)
    return SimpleNamespace(client=client, index=index)


@pytest.fixture
def valid_jwt_factory():
    def factory(role: str) -> str:
        return create_access_token({"sub": USER_FOR_ROLE[role], "role": role})

    return factory


@pytest.fixture
def expired_jwt() -> str:
    payload = {
        "sub": "alice",
        "role": "finance",
        "exp": datetime.now(timezone.utc) - timedelta(minutes=5),
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


@pytest.fixture
def auth_header(valid_jwt_factory):
    return lambda role: {"Authorization": f"Bearer {valid_jwt_factory(role)}"}


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)
