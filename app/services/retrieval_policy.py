"""Per-collection retrieval strategy.

This module only decides HOW to search a collection. It makes no access
control decisions — the collection list is filtered by role before any
policy is looked up.
"""

from dataclasses import dataclass
from typing import Literal

from app.config import settings


@dataclass
class RetrievalPolicy:
    collection_name: str
    strategy: Literal["semantic", "hybrid"]
    dense_top_k: int
    bm25_top_k: int


# HR and finance contain exact identifiers (employee IDs, names, figures,
# vendor names) that lexical BM25 matches far better than embeddings.
# Depths come from settings (DENSE_TOP_K / BM25_TOP_K, both default 15).
_DENSE = settings.DENSE_TOP_K
_BM25 = settings.BM25_TOP_K

RETRIEVAL_POLICY: dict[str, RetrievalPolicy] = {
    "finance": RetrievalPolicy("finance", "hybrid", _DENSE, _BM25),
    "hr_docs": RetrievalPolicy("hr_docs", "hybrid", _DENSE, _BM25),
    "marketing": RetrievalPolicy("marketing", "semantic", _DENSE, 0),
    "engineering": RetrievalPolicy("engineering", "semantic", _DENSE, 0),
    "general": RetrievalPolicy("general", "semantic", _DENSE, 0),
}


def get_policy(collection_name: str) -> RetrievalPolicy:
    return RETRIEVAL_POLICY.get(
        collection_name,
        RetrievalPolicy(collection_name, "semantic", _DENSE, 0),
    )
