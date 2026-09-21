import threading
import time
from functools import lru_cache
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from app.config import settings
from app.services.bm25_retriever import bm25_retriever
from app.services.cache import semantic_cache
from app.services.query_rewriter import needs_rewriting, rewrite_query
from app.services.reranker import reranker
from app.services.retrieval_policy import get_policy
from app.services.rrf import fuse_across_collections, reciprocal_rank_fusion
from app.services.session_store import session_store
from app.utils.embeddings import get_embeddings
from app.utils.logger import RequestContext, get_logger

logger = get_logger(__name__)

HR_COLLECTION = "hr_docs"
ALL_COLLECTIONS = ["finance", "marketing", HR_COLLECTION, "engineering", "general"]
# The exact refusal the system prompt tells the LLM to give.
DECLINED_ANSWER = "I don't have access to that information."


def is_declined_answer(answer: str) -> bool:
    """True when the LLM gave the prompt's refusal instead of an answer.

    Tolerates the variations models actually produce: curly apostrophes,
    surrounding quotes/whitespace and a missing trailing period.
    """
    normalized = answer.strip().replace("\u2019", "'").strip("\"'` ").rstrip(".").strip().lower()
    return normalized == DECLINED_ANSWER.rstrip(".").lower()


NO_RELEVANT_CONTEXT_ANSWER = (
    "I was unable to find sufficiently relevant information "
    "in your authorized documents to answer this question."
)

_vectorstores: dict[str, Chroma] = {}
_vectorstores_lock = threading.Lock()


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    """Groq chat model shared by query rewriting and answer generation."""
    return ChatGroq(
        groq_api_key=settings.GROQ_API_KEY,
        model_name=settings.GROQ_MODEL,
        temperature=0.2,
    )


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


def query_rag(
    question: str,
    role: str,
    session_id: str | None = None,
    ctx: RequestContext | None = None,
) -> dict[str, Any]:
    ctx = ctx or RequestContext()
    cleaned_question = question.strip()
    if not cleaned_question:
        return {"answer": "I don't have access to that information", "sources": []}

    # RBAC: authorized collections are decided here, before any retrieval.
    collections = get_collections_for_role(role)
    ctx.user_role = role
    ctx.authorized_collections = list(collections)
    ctx.original_query = cleaned_question

    history = (
        session_store.get_history(session_id, max_messages=settings.CONVERSATION_HISTORY_LENGTH)
        if session_id
        else []
    )

    retrieval_query = cleaned_question
    if settings.QUERY_REWRITING_ENABLED and needs_rewriting(cleaned_question, history):
        t0 = time.perf_counter()
        retrieval_query = rewrite_query(cleaned_question, history, get_llm())
        ctx.latency_ms["rewrite"] = (time.perf_counter() - t0) * 1000
        ctx.query_rewritten = True
        ctx.rewritten_query = retrieval_query
        logger.debug("Query rewritten for retrieval")
    else:
        ctx.query_rewritten = False

    # Semantic cache. `collections` is already RBAC-filtered, and the cache
    # namespace is built from role + those collections, so entries are never
    # shared across roles. The lookup uses the standalone retrieval query:
    # for self-contained questions that is the original question (no LLM call
    # has happened yet); a follow-up like "what about Q4?" is only cached in
    # its rewritten form, so one conversation's context cannot leak into
    # another's cached answer.
    t0 = time.perf_counter()
    cached = semantic_cache.get(retrieval_query, role, collections)
    ctx.latency_ms["cache_lookup"] = (time.perf_counter() - t0) * 1000
    if cached is not None:
        ctx.cache_hit = True
        if session_id:
            session_store.add_message(session_id, "user", cleaned_question)
            session_store.add_message(session_id, "assistant", cached.get("answer", ""))
        return cached
    ctx.cache_hit = False

    # retrieval_query drives ChromaDB, BM25 and the reranker; RBAC scope
    # (`collections`) was fixed above from the role and is not affected.
    t0 = time.perf_counter()
    candidates = retrieve_for_collections(retrieval_query, collections, get_embeddings(), ctx)
    ctx.latency_ms["retrieval"] = (time.perf_counter() - t0) * 1000

    ctx.reranker_candidate_count = len(candidates)

    if not candidates:
        return {"answer": "I don't have access to that information", "sources": []}

    t0 = time.perf_counter()
    reranked_docs, scores = reranker.rerank(
        query=retrieval_query,
        candidates=candidates,
        top_k=settings.RERANKER_FINAL_TOP_K,
        threshold=settings.RERANKER_SCORE_THRESHOLD,
    )
    ctx.latency_ms["rerank"] = (time.perf_counter() - t0) * 1000
    ctx.reranker_scores = scores
    ctx.final_context_count = len(reranked_docs)

    # Nothing cleared the relevance threshold: answer safely without the LLM.
    if not reranked_docs:
        return {"answer": NO_RELEVANT_CONTEXT_ANSWER, "sources": []}

    # Only threshold-passing, reranked, authorized chunks reach the LLM.
    relevant_docs = reranked_docs
    sources = sorted({doc.metadata.get("source", "unknown") for doc in relevant_docs})

    context = "\n\n".join(
        [
            f"Source: {doc.metadata.get('source', 'unknown')}\nContent:\n{doc.page_content}"
            for doc in relevant_docs
        ]
    )
    if not context.strip():
        return {"answer": "I don't have access to that information", "sources": []}

    llm = get_llm()

    system_prompt = (
        "You are FinSolve Technologies' secure enterprise assistant. "
        "Answer ONLY using the provided context. "
        "Do NOT guess, infer, or use any outside knowledge. "
        "If the user's question cannot be clearly and directly answered "
        "using the provided context, respond exactly with: "
        "\"I don't have access to that information.\" "
        "Always cite the source documents used in your answer."
    )

    # The original question is what the user asked and what gets answered.
    # When it was rewritten, the standalone form is added so the model can
    # resolve references like "it" / "that" without seeing prior turns.
    question_block = f"Question: {cleaned_question}"
    if ctx.query_rewritten and retrieval_query != cleaned_question:
        question_block += f"\n(Interpreted as: {retrieval_query})"
    human_prompt = f"{question_block}\n\nContext:\n{context}"
    t0 = time.perf_counter()
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
    ctx.latency_ms["llm"] = (time.perf_counter() - t0) * 1000

    answer_text = response.content if isinstance(response.content, str) else str(response.content)
    if not answer_text.strip():
        answer_text = DECLINED_ANSWER

    declined = is_declined_answer(answer_text)
    if declined:
        # The LLM judged the context insufficient, so no document was used:
        # listing the retrieved sources next to a refusal would be misleading.
        sources = []

    if session_id:
        session_store.add_message(session_id, "user", cleaned_question)
        session_store.add_message(session_id, "assistant", answer_text)

    result = {"answer": answer_text, "sources": sources}
    # Refusals are not cached, like the no-relevant-context answer, so a
    # retrieval fix or re-ingest is not masked by a stored refusal for CACHE_TTL.
    if not declined:
        semantic_cache.set(retrieval_query, role, collections, result)
    return result
