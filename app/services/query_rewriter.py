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


MAX_REWRITE_GROWTH = 6  # a rewrite is a short question, not an answer or an essay
_REWRITE_PREFIX = re.compile(r"^(rewritten question|rewritten|question)\s*:\s*", re.IGNORECASE)


def _clean_rewrite(raw: str, question: str) -> str:
    """The rewritten question, or the original when the model's output is not a usable question.

    Models sometimes wrap the question in quotes or a label, or answer instead of
    rewriting. Only the first line is used, and a result far longer than the
    original is treated as an answer, not a rewrite.
    """
    lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]
    if not lines:
        return question
    rewritten = _REWRITE_PREFIX.sub("", lines[0]).strip().strip("\"'`").strip()
    if not rewritten or len(rewritten) > max(200, MAX_REWRITE_GROWTH * len(question)):
        return question
    return rewritten


def rewrite_query(question: str, history: list[ChatMessage], llm: BaseChatModel) -> str:
    history_text = "\n".join(f"{msg['role'].upper()}: {msg['content']}" for msg in history[-6:])

    system_prompt = (
        "You rewrite a follow-up question so it can be understood without the conversation history. "
        "Replace pronouns and references (it, that, those, the same, the previous one) with the exact "
        "subject they refer to in the history. "
        "Keep every name, number, date, period and unit from the follow-up question and from the history "
        "that the question depends on, exactly as written. "
        "Do not answer the question, do not add facts, assumptions or new topics, and do not "
        "broaden or narrow what is asked. "
        "If the question is already self-contained, return it exactly as is. "
        "Return ONLY the rewritten question on a single line. "
        "No explanation, no preamble, no quotation marks."
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
        return _clean_rewrite(message_text(response), question)
    except Exception:
        # Fallback to original question on any failure
        logger.warning("Query rewriting failed, using original question", exc_info=True)
        return question
