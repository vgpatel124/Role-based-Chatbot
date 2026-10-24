"""The chat model behind query rewriting and answer generation, and reading its output.

LLM_PROVIDER picks Groq (default) or Anthropic. Both are built here so the rest of the
app never names a provider, and so the differences between them live in one place:

* Claude returns message content as a list of blocks (text, and thinking blocks when the
  model thinks), where Groq returns a string. `message_text` reads both.
* Claude models reject sampling parameters such as temperature (the API answers 400),
  so none is sent to Anthropic; Groq keeps its temperature.
"""

from typing import Any

from langchain_core.language_models import BaseChatModel

from app.config import settings

# Answers are short, but adaptive thinking spends output tokens too, and the Anthropic API
# requires an explicit ceiling.
ANTHROPIC_MAX_TOKENS = 4096

# Anthropic stop reasons that mean the reply ended on its own (not cut off or refused).
FINISHED_STOP_REASONS = ("end_turn", "stop_sequence")


def llm_model_name() -> str:
    return settings.ANTHROPIC_MODEL if settings.LLM_PROVIDER == "anthropic" else settings.GROQ_MODEL


def missing_llm_credential() -> str | None:
    """Name of the API key the active provider needs but does not have, else None."""
    if settings.LLM_PROVIDER == "anthropic":
        return None if settings.ANTHROPIC_API_KEY else "ANTHROPIC_API_KEY"
    return None if settings.GROQ_API_KEY else "GROQ_API_KEY"


def build_chat_model() -> BaseChatModel:
    missing = missing_llm_credential()
    if missing:
        raise RuntimeError(f"LLM_PROVIDER={settings.LLM_PROVIDER} needs {missing} (environment or .env)")

    if settings.LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.ANTHROPIC_MODEL,
            max_tokens=ANTHROPIC_MAX_TOKENS,
            anthropic_api_key=settings.ANTHROPIC_API_KEY,
        )

    from langchain_groq import ChatGroq

    return ChatGroq(groq_api_key=settings.GROQ_API_KEY, model_name=settings.GROQ_MODEL, temperature=0.2)


def message_text(message: Any) -> str:
    """The text of a chat model reply: a string as is, or the text blocks of a block list.

    Thinking blocks and anything else that is not text are left out. A reply with no text
    (for example a safety refusal) gives an empty string.
    """
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def describe_error(exc: Exception) -> str:
    """Error class and HTTP status only: provider messages are kept out of logs and reports."""
    status = getattr(exc, "status_code", None)
    return type(exc).__name__ + (f" (HTTP {status})" if status else "")


def preflight_chat_model() -> str | None:
    """None when the chatbot's LLM is usable, otherwise why not.

    With the default Groq provider there is nothing to check up front (a failure shows up
    on the first call). With Anthropic the model's availability is checked for free and the
    exact configuration of the chatbot's model gets one tiny call, so a rejected key, an
    unavailable model or a rejected request shape fails before any evaluation work starts.
    """
    if settings.LLM_PROVIDER != "anthropic":
        return None
    model = settings.ANTHROPIC_MODEL
    try:
        import anthropic

        anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY).models.retrieve(model)  # free
    except Exception as exc:
        return f"chatbot model {model} could not be confirmed: {describe_error(exc)}"
    try:
        reply = build_chat_model().invoke("Reply with the single word OK.")
    except Exception as exc:
        return f"test call to chatbot model {model} failed: {describe_error(exc)}"
    if (getattr(reply, "response_metadata", None) or {}).get("stop_reason") not in FINISHED_STOP_REASONS:
        return f"test call to chatbot model {model} did not finish normally"
    if not message_text(reply).strip():
        return f"test call to chatbot model {model} returned no text"
    return None
