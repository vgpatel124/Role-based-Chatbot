"""Conditional rewriting of context-dependent follow-up questions.

Rewriting costs an extra LLM call, so it only runs when there is history
AND the question looks like it depends on it (pronouns, "what about ...").
Rewriting only changes the retrieval query; RBAC is decided beforehand
from the user's role, so a rewritten query can never widen access.
"""

import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.services.session_store import ChatMessage
from app.utils.llm import message_text
from app.utils.logger import get_logger
from app.utils.tracing import llm_invoke

logger = get_logger(__name__)

CONTEXT_DEPENDENCY_SIGNALS = [
    "it", "that", "this", "they", "those", "there", "same",
    "above", "previous", "mentioned", "compared to", "the same",
]

CONTEXT_DEPENDENCY_STARTERS = [
    "what about", "how about", "and what", "but what",
    "and how", "but how", "what if", "and if",
]

_NON_WORD = re.compile(r"[^\w\s]")
_WHITESPACE = re.compile(r"\s+")


def _normalize(question: str) -> str:
    # Punctuation becomes whitespace so "And that?" still matches " that ".
    return _WHITESPACE.sub(" ", _NON_WORD.sub(" ", question.lower())).strip()


def needs_rewriting(question: str, history: list[ChatMessage]) -> bool:
    if not history:
        return False
    q_lower = _normalize(question)
    for signal in CONTEXT_DEPENDENCY_SIGNALS:
        if f" {signal} " in f" {q_lower} ":
            return True
    for starter in CONTEXT_DEPENDENCY_STARTERS:
        if q_lower.startswith(starter):
            return True
    return False


def rewrite_query(question: str, history: list[ChatMessage], llm: BaseChatModel) -> str:
    history_text = "\n".join(f"{msg['role'].upper()}: {msg['content']}" for msg in history[-6:])

    system_prompt = (
        "You are a query rewriting assistant. "
        "Given a conversation history and a follow-up question, "
        "rewrite the follow-up question to be completely "
        "self-contained so it can be understood without the "
        "conversation history. If the question is already "
        "self-contained, return it exactly as is. "
        "Return ONLY the rewritten question, nothing else. "
        "No explanation, no preamble."
    )

    human_prompt = (
        f"Conversation history:\n{history_text}\n\n"
        f"Follow-up question: {question}\n\n"
        f"Rewritten question:"
    )

    try:
        response = llm_invoke(
            llm, [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)], "rewrite-query"
        )
        rewritten = message_text(response).strip()
        if not rewritten:
            return question
        return rewritten
    except Exception:
        # Fallback to original question on any failure
        logger.warning("Query rewriting failed, using original question", exc_info=True)
        return question
