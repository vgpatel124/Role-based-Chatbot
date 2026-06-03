from typing import Any

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from app.config import settings

HR_COLLECTION = "hr_docs"
ALL_COLLECTIONS = ["finance", "marketing", HR_COLLECTION, "engineering", "general"]
TOP_K = 8


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


def _retrieve_relevant_docs(question: str, collections: list[str], top_k: int = TOP_K) -> list[tuple[Any, float]]:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    scored_docs: list[tuple[Any, float]] = []

    for collection in collections:
        vectorstore = Chroma(
            persist_directory=settings.CHROMA_DB_PATH,
            collection_name=collection,
            embedding_function=embeddings,
        )
        try:
            docs_with_scores = vectorstore.similarity_search_with_relevance_scores(question, k=top_k)
            scored_docs.extend(docs_with_scores)
        except Exception:
            docs = vectorstore.similarity_search(question, k=top_k)
            scored_docs.extend((doc, 0.0) for doc in docs)

    scored_docs.sort(key=lambda item: item[1], reverse=True)
    return scored_docs[:top_k]


def query_rag(question: str, role: str) -> dict[str, Any]:
    cleaned_question = question.strip()
    if not cleaned_question:
        return {"answer": "I don't have access to that information", "sources": []}

    collections = get_collections_for_role(role)
    scored_docs = _retrieve_relevant_docs(cleaned_question, collections, top_k=TOP_K)

    if not scored_docs:
        return {"answer": "I don't have access to that information", "sources": []}

    relevant_docs = [doc for doc, _ in scored_docs]
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
