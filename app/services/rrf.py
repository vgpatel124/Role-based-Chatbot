"""Reciprocal Rank Fusion (Cormack et al., 2009).

score(d) = sum over ranked lists L containing d of 1 / (k + rank_L(d)),
with rank starting at 1. Only ranks are used, so dense cosine scores and
BM25 scores never need to be put on a common scale.
"""

import hashlib

from langchain_core.documents import Document


def content_hash(doc: Document) -> str:
    return hashlib.md5(doc.page_content.encode()).hexdigest()


def reciprocal_rank_fusion(
    dense_results: list[Document],
    bm25_results: list[Document],
    k: int = 60,
    top_n: int = 15,
) -> list[Document]:
    scores: dict[str, float] = {}
    dense_ranks: dict[str, int] = {}
    bm25_ranks: dict[str, int] = {}
    docs: dict[str, Document] = {}

    for i, doc in enumerate(dense_results):
        key = content_hash(doc)
        if key in dense_ranks:
            continue  # duplicate chunk within the same list: keep best rank
        dense_ranks[key] = i + 1
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + i + 1)
        docs.setdefault(key, doc)

    for j, doc in enumerate(bm25_results):
        key = content_hash(doc)
        if key in bm25_ranks:
            continue
        bm25_ranks[key] = j + 1
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + j + 1)
        docs.setdefault(key, doc)

    ranked_keys = sorted(scores, key=lambda key: scores[key], reverse=True)[:top_n]

    fused: list[Document] = []
    for key in ranked_keys:
        dense_rank = dense_ranks.get(key)
        bm25_rank = bm25_ranks.get(key)
        if dense_rank is not None and bm25_rank is not None:
            source = "both"
        elif dense_rank is not None:
            source = "dense"
        else:
            source = "bm25"
        original = docs[key]
        # New Document so inputs (possibly cached objects) are never mutated.
        fused.append(
            Document(
                page_content=original.page_content,
                metadata={
                    **original.metadata,
                    "rrf_score": scores[key],
                    "dense_rank": dense_rank,
                    "bm25_rank": bm25_rank,
                    "retrieval_source": source,
                },
            )
        )
    return fused


def fuse_across_collections(
    ranked_lists: list[list[Document]],
    k: int = 60,
    top_n: int = 15,
) -> list[Document]:
    """RRF over any number of per-collection result lists (c_level case).

    Each list keeps its per-collection metadata (dense_rank, bm25_rank,
    retrieval_source); the global fused score is stored as "global_rrf_score".
    """
    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}
    for ranked in ranked_lists:
        seen: set[str] = set()
        for i, doc in enumerate(ranked):
            key = content_hash(doc)
            if key in seen:
                continue
            seen.add(key)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + i + 1)
            docs.setdefault(key, doc)

    ranked_keys = sorted(scores, key=lambda key: scores[key], reverse=True)[:top_n]
    return [
        Document(
            page_content=docs[key].page_content,
            metadata={**docs[key].metadata, "global_rrf_score": scores[key]},
        )
        for key in ranked_keys
    ]
