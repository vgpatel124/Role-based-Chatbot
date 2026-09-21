from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    session_id: str | None = None

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Question must not be empty.")
        if len(cleaned) > 1000:
            raise ValueError("Question must be at most 1000 characters.")
        return cleaned


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    role: str
    session_id: str
    cache_hit: bool = False
