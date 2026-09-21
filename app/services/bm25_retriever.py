"""Lexical BM25 retrieval over chunks persisted by app/ingest.py.

Reads {CHROMA_DB_PATH}/bm25_docs/{collection}.json once per process and
keeps a BM25Okapi index in memory. Missing or unreadable files degrade to
an empty result so the caller falls back to semantic-only retrieval.
"""

import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_TOKEN_PATTERN = re.compile(r"\w+")

# Small stopword list so question scaffolding ("what is the ...") does not
# dominate lexical scoring. Identifiers like FINEMP1005 are kept intact.
_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "did", "do",
        "does", "for", "from", "has", "have", "how", "i", "in", "is", "it",
        "me", "much", "of", "on", "or", "show", "tell", "that", "the", "their",
        "this", "to", "was", "were", "what", "when", "where", "which", "who",
        "why", "with", "you", "your",
    }
)


def tokenize(text: str) -> list[str]:
    return [tok for tok in _TOKEN_PATTERN.findall(text.lower()) if tok not in _STOPWORDS]


@dataclass
class _BM25Index:
    bm25: BM25Okapi
    documents: list[Document]


class BM25RetrieverService:
    def __init__(self) -> None:
        # None marks a collection whose JSON was missing/invalid, so we do not
        # retry the disk read on every request.
        self._indexes: dict[str, _BM25Index | None] = {}
        self._lock = threading.Lock()

    def _json_path(self, collection_name: str) -> Path:
        return Path(settings.CHROMA_DB_PATH) / "bm25_docs" / f"{collection_name}.json"

    def _load(self, collection_name: str) -> _BM25Index | None:
        path = self._json_path(collection_name)
        if not path.exists():
            logger.warning(
                "BM25 JSON not found for %s, falling back to semantic only",
                collection_name,
            )
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                records = json.load(f)
            documents = [
                Document(
                    page_content=record["page_content"],
                    metadata={
                        "source": record.get("metadata", {}).get("source", "unknown"),
                        "role": record.get("metadata", {}).get("role"),
                    },
                )
                for record in records
                if record.get("page_content")
            ]
            if not documents:
                logger.warning(
                    "BM25 JSON for %s is empty, falling back to semantic only",
                    collection_name,
                )
                return None
            index = _BM25Index(
                bm25=BM25Okapi([tokenize(doc.page_content) for doc in documents]),
                documents=documents,
            )
            logger.info("BM25 index loaded: %s (%d chunks)", collection_name, len(documents))
            return index
        except Exception:
            logger.exception(
                "Failed to load BM25 JSON for %s, falling back to semantic only",
                collection_name,
            )
            return None

    def _get_index(self, collection_name: str) -> _BM25Index | None:
        with self._lock:
            if collection_name not in self._indexes:
                self._indexes[collection_name] = self._load(collection_name)
            return self._indexes[collection_name]

    def retrieve(self, question: str, collection_name: str, top_k: int) -> list[Document]:
        if top_k <= 0:
            return []
        try:
            index = self._get_index(collection_name)
            if index is None:
                return []
            query_tokens = tokenize(question)
            if not query_tokens:
                return []

            scores = index.bm25.get_scores(query_tokens)
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            results: list[Document] = []
            for i in ranked[:top_k]:
                # Zero score means no query term overlap — not a real match.
                if scores[i] <= 0:
                    break
                src = index.documents[i]
                # Copy so per-request metadata never leaks into the cached index.
                results.append(
                    Document(
                        page_content=src.page_content,
                        metadata={**src.metadata, "bm25_score": float(scores[i])},
                    )
                )
            logger.debug("BM25 %s: %d results", collection_name, len(results))
            return results
        except Exception:
            logger.exception("BM25 retrieval failed for %s", collection_name)
            return []


# Module-level singleton — indexes are loaded once per process.
bm25_retriever = BM25RetrieverService()
