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
from app.services.reference_resolver import REFERENCE_FIELDS, resolve_references
from app.services.reranker import reranker
from app.services.retrieval_policy import get_policy
from app.services.rrf import fuse_across_collections, reciprocal_rank_fusion
from app.services.session_store import session_store
from app.utils.embeddings import get_embeddings
from app.utils.logger import RequestContext, get_logger
from app.utils.tracing import llm_invoke, observation, score

logger = get_logger(__name__)

HR_COLLECTION = "hr_docs"
ALL_COLLECTIONS = ["finance", "marketing", HR_COLLECTION, "engineering", "general"]
# The exact refusal the system prompt tells the LLM to give.
DECLINED_ANSWER = "I don't have access to that information."

SYSTEM_PROMPT = (
    "You are FinSolve Technologies' secure enterprise assistant. "
    "Answer ONLY using the provided context. "
    "Do NOT guess, infer, or use any outside knowledge. "
    "If the user's question cannot be clearly and directly answered "
    "using the provided context, respond exactly with: "
    "\"I don't have access to that information.\" "
    "If different source documents give different values for the same figure, "
    "do not choose one silently: report each value with the document it comes "
    "from and state that the documents differ. "
    "Always cite the source documents used in your answer."
)


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

        with observation(
            "search-collection",
            as_type="retriever",
            metadata={"collection": collection_name, "strategy": policy.strategy},
        ) as step:
            try:
                vectorstore = _get_vectorstore(collection_name, embeddings)
                dense_docs = vectorstore.similarity_search(question, k=policy.dense_top_k)
            except Exception:
                logger.exception("Dense retrieval failed for %s", collection_name)
                dense_docs = []
            ctx.dense_count += len(dense_docs)

            bm25_docs: list[Document] = []
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

            step.update(
                metadata={
                    "collection": collection_name,
                    "strategy": policy.strategy,
                    "dense_results": len(dense_docs),
                    "bm25_results": len(bm25_docs),
                    "fused_results": len(collection_results),
                }
            )

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
    return_contexts: bool = False,
) -> dict[str, Any]:
    """Answer `question` for `role`.

    return_contexts=True adds the chunk texts the LLM saw under "contexts",
    for offline evaluation (RAGAS). The API never sets it: document text must
    not reach responses or logs.

    One call is one Langfuse trace (see app/utils/tracing.py), keyed to the
    request id so a request can be found from its X-Request-ID.
    """
    ctx = ctx or RequestContext()
    with observation(
        "answer-question",
        as_type="chain",
        root=True,
        trace_seed=ctx.request_id or None,
        input=question.strip(),
        metadata={"role": role},
    ) as span:
        ctx.trace_id = span.trace_id or ""
        result = _answer_question(question, role, session_id, ctx, return_contexts)
        _record_outcome(span, role, ctx, result)
        return result


def _record_outcome(span, role: str, ctx: RequestContext, result: dict[str, Any]) -> None:
    """Answer, routing facts and quality signals for the trace. Text only goes in
    `output`, which is withheld unless content capture is on; everything else is
    names, counts and numbers."""
    answer = result["answer"]
    span.update(
        output=answer,
        metadata={
            "role": role,
            "authorized_collections": ctx.authorized_collections,
            "sources": result["sources"],
            "cache_hit": ctx.cache_hit,
            "query_rewritten": ctx.query_rewritten,
            "retrieval_strategy": ctx.retrieval_strategy,
            "retrieval_version": settings.RETRIEVAL_VERSION,
            "final_context_count": ctx.final_context_count,
            "latency_ms": {name: round(value, 1) for name, value in ctx.latency_ms.items()},
        },
    )
    score("cache_hit", ctx.cache_hit)
    score("answer_declined", is_declined_answer(answer))
    score("no_relevant_context", answer == NO_RELEVANT_CONTEXT_ANSWER)
    if ctx.reranker_scores:
        score("reranker_top_score", max(ctx.reranker_scores))


def _answer_question(
    question: str,
    role: str,
    session_id: str | None,
    ctx: RequestContext,
    return_contexts: bool,
) -> dict[str, Any]:
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
        retrieval_query = rewrite_query(cleaned_question, history, get_llm())  # traced as "rewrite-query"
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
    with observation("lookup-cache", as_type="retriever") as step:
        cached = semantic_cache.get(retrieval_query, role, collections)
        step.update(metadata={"hit": cached is not None})
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
    with observation("retrieve-context", as_type="retriever", metadata={"collections": collections}) as step:
        candidates = retrieve_for_collections(retrieval_query, collections, get_embeddings(), ctx)
        step.update(
            metadata={
                "collections": collections,
                "strategy": ctx.retrieval_strategy,
                "dense_results": ctx.dense_count,
                "bm25_results": ctx.bm25_count,
                "candidates": len(candidates),
            }
        )
    ctx.latency_ms["retrieval"] = (time.perf_counter() - t0) * 1000

    ctx.reranker_candidate_count = len(candidates)

    if not candidates:
        return {"answer": "I don't have access to that information", "sources": []}

    t0 = time.perf_counter()
    with observation(
        "rerank-candidates",
        metadata={"candidates": len(candidates), "threshold": settings.RERANKER_SCORE_THRESHOLD},
    ) as step:
        reranked_docs, scores = reranker.rerank(
            query=retrieval_query,
            candidates=candidates,
            top_k=settings.RERANKER_FINAL_TOP_K,
            threshold=settings.RERANKER_SCORE_THRESHOLD,
        )
        step.update(
            metadata={
                "candidates": len(candidates),
                "threshold": settings.RERANKER_SCORE_THRESHOLD,
                "kept": len(reranked_docs),
                "scores": [round(float(value), 4) for value in scores],
            }
        )
    ctx.latency_ms["rerank"] = (time.perf_counter() - t0) * 1000
    ctx.reranker_scores = scores
    ctx.final_context_count = len(reranked_docs)

    # Nothing cleared the relevance threshold: answer safely without the LLM.
    if not reranked_docs:
        return {"answer": NO_RELEVANT_CONTEXT_ANSWER, "sources": []}

    # Only threshold-passing, reranked, authorized chunks reach the LLM, plus
    # rows they point to (an employee's manager), which are looked up only in
    # the same authorized collection (see reference_resolver).
    resolved_docs: list[Document] = []
    if any(name in REFERENCE_FIELDS for name in collections):  # only HR rows point to other rows
        with observation("resolve-references", as_type="retriever") as step:
            resolved_docs = resolve_references(reranked_docs, collections)
            step.update(metadata={"rows_added": len(resolved_docs)})
    relevant_docs = reranked_docs + resolved_docs
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

    # The original question is what the user asked and what gets answered.
    # When it was rewritten, the standalone form is added so the model can
    # resolve references like "it" / "that" without seeing prior turns.
    question_block = f"Question: {cleaned_question}"
    if ctx.query_rewritten and retrieval_query != cleaned_question:
        question_block += f"\n(Interpreted as: {retrieval_query})"
    human_prompt = f"{question_block}\n\nContext:\n{context}"
    t0 = time.perf_counter()
    response = llm_invoke(
        llm, [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=human_prompt)], "generate-answer"
    )
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
    if return_contexts:
        return {**result, "contexts": [doc.page_content for doc in relevant_docs]}
    return result
