"""Structured logging and per-request observability context.

NEVER log JWT tokens, passwords/hashes, raw document page_content, or
employee PII (salary, DOB, email). RequestContext deliberately holds only
counts, scores, role/collection names, and the user's own query text.
"""

import json
import logging
import sys
from dataclasses import asdict, dataclass, field

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"

# Queries are user input and could be long; cap what lands in the log line.
_MAX_QUERY_LOG_CHARS = 300

_ROOT_LOGGER_NAME = "app"
_configured = False


def _configure_root() -> None:
    global _configured
    if _configured:
        return
    root = logging.getLogger(_ROOT_LOGGER_NAME)
    root.setLevel(logging.INFO)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        root.addHandler(handler)
    # Keep our records out of uvicorn's/root handlers to avoid duplicate lines.
    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the shared "app" hierarchy (e.g. app.rag)."""
    _configure_root()
    if name != _ROOT_LOGGER_NAME and not name.startswith(f"{_ROOT_LOGGER_NAME}."):
        name = f"{_ROOT_LOGGER_NAME}.{name}"
    return logging.getLogger(name)


@dataclass
class RequestContext:
    request_id: str = ""
    session_id: str = ""
    user_role: str = ""
    authorized_collections: list[str] = field(default_factory=list)
    cache_hit: bool = False
    query_rewritten: bool = False
    original_query: str = ""
    rewritten_query: str | None = None
    retrieval_strategy: str = ""  # "semantic" or "hybrid"
    dense_count: int = 0
    bm25_count: int = 0
    rrf_candidate_count: int = 0
    reranker_candidate_count: int = 0
    final_context_count: int = 0
    reranker_scores: list[float] = field(default_factory=list)
    latency_ms: dict[str, float] = field(default_factory=dict)


def _truncate(text: str | None) -> str | None:
    if text is None or len(text) <= _MAX_QUERY_LOG_CHARS:
        return text
    return text[:_MAX_QUERY_LOG_CHARS] + "…"


def log_request_context(ctx: RequestContext) -> None:
    """Emit the full request context as one JSON INFO line."""
    payload = asdict(ctx)
    payload["original_query"] = _truncate(ctx.original_query)
    payload["rewritten_query"] = _truncate(ctx.rewritten_query)
    payload["reranker_scores"] = [round(float(s), 4) for s in ctx.reranker_scores]
    payload["latency_ms"] = {k: round(float(v), 2) for k, v in ctx.latency_ms.items()}
    get_logger("request").info(
        "request_summary %s", json.dumps(payload, ensure_ascii=False, default=str)
    )
