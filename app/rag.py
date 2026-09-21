import threading
import time
from functools import lru_cache
from typing import Any

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from app.config import settings
from app.services.bm25_retriever import bm25_retriever
from app.services.retrieval_policy import get_policy
from app.services.rrf import fuse_across_collections, reciprocal_rank_fusion
from app.utils.logger import RequestContext, get_logger

logger = get_logger(__name__)

HR_COLLECTION = "hr_docs"
ALL_COLLECTIONS = ["finance", "marketing", HR_COLLECTION, "engineering", "general"]
# Interim context size until the cross-encoder reranker (Phase 5) selects it.
TOP_K = 8

_vectorstores: dict[str, Chroma] = {}
_vectorstores_lock = threading.Lock()


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Embedding model shared by all requests (loaded once per process)."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def _get_vectorstore(collection_name: str, embeddings) -> Chroma:
    with _vectorstores_lock:
        if collection_name not in _vectorstores:
            _vectorstores[collection_name] = Chroma(
                persist_directory=settings.CHROMA_DB_PATH,
                collection_name=collection_name,
                embedding_function=embeddings,
            )
        return _vectorstores[collection_name]


def get_collections_for_role(role: str) -> list[str]:
    role_map = {
        "finance": ["finance"],
        "marketing": ["marketing"],
        "hr": [HR_COLLECTION],
        "engineering": ["engineering"],
        "c_level": ALL_COLLECTIONS,
        "employee": ["general"],
    }
    return role_map.get(role, ["general"])


def retrieve_for_collections(
    question: str,
    collections: list[str],
    embeddings,
    ctx: RequestContext,
) -> list[Document]:
    """Retrieve candidates from exactly the given collections.

    `collections` MUST already be filtered by RBAC (get_collections_for_role).
    This function makes no access control decisions and never searches a
    collection that is not in the list it receives.
    """
    per_collection_results: list[list[Document]] = []
    strategies: set[str] = set()

    for collection_name in collections:
        policy = get_policy(collection_name)
        strategies.add(policy.strategy)

        try:
            vectorstore = _get_vectorstore(collection_name, embeddings)
            dense_docs = vectorstore.similarity_search(question, k=policy.dense_top_k)
        except Exception:
            logger.exception("Dense retrieval failed for %s", collection_name)
            dense_docs = []
        ctx.dense_count += len(dense_docs)

        if policy.strategy == "hybrid":
            bm25_docs = bm25_retriever.retrieve(question, collection_name, policy.bm25_top_k)
            ctx.bm25_count += len(bm25_docs)
            collection_results = reciprocal_rank_fusion(
                dense_docs,
                bm25_docs,
                k=settings.RRF_K,
                top_n=settings.RRF_CANDIDATE_COUNT,
            )
        else:
            collection_results = dense_docs

        logger.debug(
            "Retrieved %s: strategy=%s dense=%d results=%d",
            collection_name,
            policy.strategy,
            len(dense_docs),
            len(collection_results),
        )
        per_collection_results.append(collection_results)

    ctx.retrieval_strategy = "hybrid" if "hybrid" in strategies else "semantic"

    if len(per_collection_results) > 1:
        # c_level: one globally ranked list across all authorized collections.
        final_results = fuse_across_collections(
            per_collection_results,
            k=settings.RRF_K,
            top_n=settings.RERANKER_CANDIDATE_COUNT,
        )
    else:
        final_results = per_collection_results[0] if per_collection_results else []

    ctx.rrf_candidate_count = len(final_results)
    return final_results[: settings.RERANKER_CANDIDATE_COUNT]


def query_rag(question: str, role: str, ctx: RequestContext | None = None) -> dict[str, Any]:
    ctx = ctx or RequestContext()
    cleaned_question = question.strip()
    if not cleaned_question:
        return {"answer": "I don't have access to that information", "sources": []}

    # RBAC: authorized collections are decided here, before any retrieval.
    collections = get_collections_for_role(role)
    ctx.user_role = role
    ctx.authorized_collections = list(collections)
    ctx.original_query = cleaned_question

    t0 = time.perf_counter()
    candidates = retrieve_for_collections(cleaned_question, collections, get_embeddings(), ctx)
    ctx.latency_ms["retrieval"] = (time.perf_counter() - t0) * 1000

    if not candidates:
        return {"answer": "I don't have access to that information", "sources": []}

    relevant_docs = candidates[:TOP_K]
    sources = sorted({doc.metadata.get("source", "unknown") for doc in relevant_docs})

    context = "\n\n".join(
        [
            f"Source: {doc.metadata.get('source', 'unknown')}\nContent:\n{doc.page_content}"
            for doc in relevant_docs
        ]
    )
    if not context.strip():
        return {"answer": "I don't have access to that information", "sources": []}

    llm = ChatGroq(
        groq_api_key=settings.GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=0.2,
    )

    system_prompt = (
        "You are FinSolve Technologies' secure enterprise assistant. "
        "Answer ONLY using the provided context. "
        "Do NOT guess, infer, or use any outside knowledge. "
        "If the user's question cannot be clearly and directly answered "
        "using the provided context, respond exactly with: "
        "\"I don't have access to that information.\" "
        "Always cite the source documents used in your answer."
    )

    human_prompt = f"Question: {cleaned_question}\n\nContext:\n{context}"
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])

    answer_text = response.content if isinstance(response.content, str) else str(response.content)
    if not answer_text.strip():
        answer_text = "I don't have access to that information"

    return {"answer": answer_text, "sources": sources}
