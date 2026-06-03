from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, field_validator

from app.auth import LoginRequest, TokenResponse, UserMeResponse, authenticate_user, create_access_token, get_current_user
from app.rag import query_rag

app = FastAPI(title="FinSolve RBAC RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)

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


@app.post("/auth/login", response_model=TokenResponse)
def login(form_data: OAuth2PasswordRequestForm = Depends()) -> TokenResponse:
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"sub": user.username, "role": user.role})
    return TokenResponse(access_token=token)


@app.post("/auth/login-json", response_model=TokenResponse)
def login_json(payload: LoginRequest) -> TokenResponse:
    user = authenticate_user(payload.username, payload.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = create_access_token({"sub": user.username, "role": user.role})
    return TokenResponse(access_token=token)


@app.get("/auth/me", response_model=UserMeResponse)
def me(user=Depends(get_current_user)) -> UserMeResponse:
    return UserMeResponse(username=user.username, role=user.role)


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, user=Depends(get_current_user)) -> ChatResponse:
    result = query_rag(question=payload.question, role=user.role)
    return ChatResponse(answer=result["answer"], sources=result["sources"], role=user.role)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}