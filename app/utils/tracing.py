"""Langfuse tracing: one trace per chat turn, with a nested span per pipeline step.

Design rules, in priority order:

1. Tracing never breaks a request. Without keys it is a no-op, and every SDK
   call here is wrapped so a Langfuse outage or SDK error only costs the trace.
2. Document and conversation text stays out of Langfuse unless
   LANGFUSE_CAPTURE_CONTENT=true. This app serves HR and finance records, and
   the logger already refuses to write them to logs, so the same rule holds for
   a third-party backend. It is enforced once, at export, by
   `_mask_otel_spans`, which drops every input/output attribute, so a call
   site (or an integration) that records a prompt cannot leak it by accident.
   Metadata set here is limited to names, counts, scores and timings.
3. Secrets and e-mail addresses are scrubbed from whatever is exported, even
   when content capture is on.

Step spans only exist inside a trace opened by `observation(..., root=True)`,
so scripts that call retrieval helpers directly do not create stray traces.
"""

import atexit
import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import Any

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Span attributes that carry prompt, document, question or answer text. Includes
# the common OpenTelemetry GenAI / OpenInference names in case an instrumentation
# library is added later.
_CONTENT_ATTRIBUTES = frozenset(
    {
        "langfuse.observation.input",
        "langfuse.observation.output",
        "langfuse.trace.input",
        "langfuse.trace.output",
        "input.value",
        "output.value",
    }
)
_CONTENT_PREFIXES = ("gen_ai.prompt", "gen_ai.completion", "llm.input_messages", "llm.output_messages")

# Identifiers that must stay readable so traces can be filtered per user/session.
_SCRUB_EXEMPT = frozenset({"user.id", "session.id"})

_SCRUBBERS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]*"), "[jwt]"),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"), "Bearer [redacted]"),
    (re.compile(r"\b(?:gsk|sk|pk)[-_][A-Za-z0-9_-]{16,}"), "[api-key]"),
    (re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"), "[email]"),
    (re.compile(r"\borg_[A-Za-z0-9]{8,}"), "org_<redacted>"),  # provider error messages name the account
]

TRACE_NAME = "chat-request"


def scrub(text: str) -> str:
    """Replace JWTs, API keys, e-mail addresses and provider account ids in `text`."""
    for pattern, replacement in _SCRUBBERS:
        text = pattern.sub(replacement, text)
    return text


def _mask_otel_spans(*, params):
    """Export-time privacy filter (Langfuse `mask_otel_spans` hook)."""
    from langfuse.types import MaskOtelSpansResult, OtelSpanPatch

    capture = settings.LANGFUSE_CAPTURE_CONTENT
    patches = {}
    for identifier, span in params.spans.items():
        delete: list[str] = []
        replace: dict[str, str] = {}
        for key, value in span.attributes.items():
            if not capture and (key in _CONTENT_ATTRIBUTES or key.startswith(_CONTENT_PREFIXES)):
                delete.append(key)
            elif isinstance(value, str) and key not in _SCRUB_EXEMPT:
                scrubbed = scrub(value)
                if scrubbed != value:
                    replace[key] = scrubbed
        if delete or replace:
            patches[identifier] = OtelSpanPatch(delete_attributes=tuple(delete), set_attributes=replace)
    return MaskOtelSpansResult(span_patches=patches) if patches else None


# --- Client lifecycle ---------------------------------------------------------------

_client: Any = None
_initialized = False
_lock = threading.Lock()


def _configured() -> bool:
    return bool(
        settings.LANGFUSE_TRACING_ENABLED and settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY
    )


def _build_client():
    # Imported lazily so the SDK is only loaded when tracing is configured, and only
    # after app.config has loaded .env (the SDK reads its settings at construction).
    from langfuse import Langfuse

    return Langfuse(
        public_key=settings.LANGFUSE_PUBLIC_KEY,
        secret_key=settings.LANGFUSE_SECRET_KEY,
        base_url=settings.LANGFUSE_BASE_URL,
        mask_otel_spans=_mask_otel_spans,
    )


def get_client():
    """The shared Langfuse client, or None when tracing is off or failed to start."""
    global _client, _initialized
    if _initialized:
        return _client
    if not _configured():
        return None
    with _lock:
        if not _initialized:
            try:
                _client = _build_client()
                atexit.register(shutdown_tracing)
                logger.info(
                    "Langfuse tracing enabled | base_url=%s | content_capture=%s",
                    settings.LANGFUSE_BASE_URL,
                    settings.LANGFUSE_CAPTURE_CONTENT,
                )
            except Exception:
                logger.warning("Langfuse tracing could not start; continuing without it", exc_info=True)
                _client = None
            _initialized = True
    return _client


def shutdown_tracing() -> None:
    """Flush queued spans and stop the exporter (call on shutdown and at script exit)."""
    global _client, _initialized
    client = _client
    if client is None:
        return
    try:
        client.shutdown()
    except Exception:
        logger.warning("Langfuse shutdown failed", exc_info=True)
    _client = None
    _initialized = False


def reset_for_tests() -> None:
    global _client, _initialized
    _client = None
    _initialized = False


# --- Observations -------------------------------------------------------------------


class _Noop:
    trace_id: str | None = None

    def update(self, **kwargs: Any) -> None:
        pass


NOOP = _Noop()


class _Observation:
    """SDK observation whose update() cannot raise into the request path."""

    def __init__(self, observation: Any) -> None:
        self._observation = observation
        self.trace_id: str | None = getattr(observation, "trace_id", None)

    def update(self, **kwargs: Any) -> None:
        try:
            self._observation.update(**kwargs)
        except Exception:
            logger.debug("Langfuse update failed", exc_info=True)


def current_trace_id() -> str | None:
    client = get_client()
    if client is None:
        return None
    try:
        return client.get_current_trace_id()
    except Exception:
        return None


@contextmanager
def observation(
    name: str,
    *,
    as_type: str = "span",
    root: bool = False,
    trace_seed: str | None = None,
    **fields: Any,
) -> Iterator[Any]:
    """Open an observation for the duration of the block.

    root=True starts a new trace (deterministic from `trace_seed` when given, so a
    request id can be turned back into a trace id). Otherwise the observation is
    created only under an already-open trace and is skipped outside one.
    Names are verb-first and fixed; run-specific values belong in metadata.
    """
    client = get_client()
    if client is None or (not root and current_trace_id() is None):
        yield NOOP
        return

    try:
        if root and trace_seed:
            fields["trace_context"] = {"trace_id": client.create_trace_id(seed=trace_seed)}
        manager = client.start_as_current_observation(as_type=as_type, name=name, **fields)
        sdk_observation = manager.__enter__()
    except Exception:
        logger.debug("Langfuse could not open observation %s", name, exc_info=True)
        yield NOOP
        return

    observation = _Observation(sdk_observation)
    try:
        yield observation
    except BaseException as exc:
        # Mark the observation failed with the exception class only. The SDK would
        # otherwise export the exception message and stack trace as span events, which
        # the masking hook cannot reach and which can echo question or document text.
        # The full detail stays in the application log.
        observation.update(level="ERROR", status_message=type(exc).__name__)
        _close(manager, name)
        raise
    else:
        _close(manager, name)


def _close(manager: Any, name: str) -> None:
    try:
        manager.__exit__(None, None, None)
    except Exception:
        logger.debug("Langfuse could not close observation %s", name, exc_info=True)


def request_attributes(*, user_id: str, session_id: str, role: str):
    """Attach user, session, role and trace name to every observation of a request.

    Wrap the call that opens the root observation. Returns a no-op context
    manager when tracing is off.
    """
    client = get_client()
    if client is None:
        return nullcontext()
    try:
        from langfuse import propagate_attributes

        return propagate_attributes(
            user_id=user_id,
            session_id=session_id,
            tags=[f"role:{role}"],
            metadata={"role": role, "content_captured": str(settings.LANGFUSE_CAPTURE_CONTENT).lower()},
            trace_name=TRACE_NAME,
        )
    except Exception:
        logger.debug("Langfuse could not set request attributes", exc_info=True)
        return nullcontext()


def _describe(error: BaseException) -> Exception:
    """An error carrying only its class (and HTTP status), never its message."""
    status = getattr(error, "status_code", None)
    return Exception(type(error).__name__ + (f" ({status})" if status else ""))


_handler_class: Any = None


def _callback_handler():
    """Langfuse's LangChain callback handler, with error messages reduced to a class name.

    Provider and library error messages can echo prompt or document text, and the SDK
    exports them as the span status, which the masking hook cannot alter.
    """
    global _handler_class
    if _handler_class is None:
        from langfuse.langchain import CallbackHandler

        class SafeCallbackHandler(CallbackHandler):
            def on_llm_error(self, error, *args, **kwargs):
                return super().on_llm_error(_describe(error), *args, **kwargs)

            def on_chain_error(self, error, *args, **kwargs):
                return super().on_chain_error(_describe(error), *args, **kwargs)

        _handler_class = SafeCallbackHandler
    return _handler_class(public_key=settings.LANGFUSE_PUBLIC_KEY)


def langchain_config(run_name: str) -> dict[str, Any] | None:
    """LangChain `config` that records the LLM call as a generation (model, tokens,
    latency) nested under the current observation; None when not tracing."""
    if current_trace_id() is None:
        return None
    try:
        return {"callbacks": [_callback_handler()], "run_name": run_name}
    except Exception:
        logger.debug("Langfuse callback handler unavailable", exc_info=True)
        return None


def llm_invoke(llm: Any, messages: list, run_name: str) -> Any:
    """`llm.invoke(messages)`, recorded as a generation named `run_name` when tracing."""
    config = langchain_config(run_name)
    return llm.invoke(messages, config=config) if config else llm.invoke(messages)


def score(name: str, value: float | bool, *, comment: str | None = None) -> None:
    """Attach a score to the current trace (booleans become BOOLEAN scores)."""
    client = get_client()
    if client is None or current_trace_id() is None:
        return
    try:
        if isinstance(value, bool):
            client.score_current_trace(name=name, value=float(value), data_type="BOOLEAN", comment=comment)
        else:
            client.score_current_trace(name=name, value=float(value), data_type="NUMERIC", comment=comment)
    except Exception:
        logger.debug("Langfuse score %s failed", name, exc_info=True)
