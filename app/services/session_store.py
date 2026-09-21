"""Per-session conversation history.

SessionStore is a Protocol, so a PostgreSQL- or Redis-backed store can
replace InMemorySessionStore later without changing any calling code.
"""

import threading
from typing import Protocol, TypedDict, runtime_checkable


class ChatMessage(TypedDict):
    role: str  # "user" or "assistant"
    content: str


@runtime_checkable
class SessionStore(Protocol):
    def get_history(self, session_id: str, max_messages: int = 6) -> list[ChatMessage]: ...

    def add_message(self, session_id: str, role: str, content: str) -> None: ...


class InMemorySessionStore:
    def __init__(self) -> None:
        self._store: dict[str, list[ChatMessage]] = {}
        self._lock = threading.Lock()

    def get_history(self, session_id: str, max_messages: int = 6) -> list[ChatMessage]:
        if max_messages <= 0:
            return []
        with self._lock:
            history = self._store.get(session_id, [])
            # Copies, so callers cannot mutate stored history.
            return [ChatMessage(role=m["role"], content=m["content"]) for m in history[-max_messages:]]

    def add_message(self, session_id: str, role: str, content: str) -> None:
        with self._lock:
            if session_id not in self._store:
                self._store[session_id] = []
            self._store[session_id].append(ChatMessage(role=role, content=content))


# Module-level singleton — import this everywhere
session_store = InMemorySessionStore()
