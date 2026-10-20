import time
import uuid

from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm

from app.auth import (
    LoginRequest,
    TokenResponse,
    UserMeResponse,
    authenticate_user,
    create_access_token,
    get_current_user,
)
from app.config import settings
from app.rag import query_rag
from app.schemas.chat import ChatRequest, ChatResponse

# Imported at module level so the reranker model, Redis connection and
# session store are created once at startup, not on the first request.
from app.services.cache import semantic_cache  # noqa: F401
from app.services.reranker import reranker  # noqa: F401
from app.services.session_store import session_store  # noqa: F401
from app.utils.embeddings import get_embeddings
from app.utils.logger import RequestContext, get_logger, log_request_context

logger = get_logger(__name__)

# Warm the shared embedding model now; otherwise the first /chat pays ~8s
# loading it inside the cache lookup.
get_embeddings()

app = FastAPI(title="FinSolve RBAC RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
def chat(payload: ChatRequest, response: Response, user=Depends(get_current_user)) -> ChatResponse:
    request_id = str(uuid.uuid4())
    response.headers["X-Request-ID"] = request_id
    session_id = payload.session_id or str(uuid.uuid4())
    ctx = RequestContext(request_id=request_id, session_id=session_id, user_role=user.role)

    started = time.perf_counter()
    try:
        # History is keyed per user, so a session_id supplied by another user
        # can never read or append to someone else's conversation.
        history_key = f"{user.username}:{session_id}"
        result = query_rag(
            question=payload.question, role=user.role, session_id=history_key, ctx=ctx
        )
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            role=user.role,
            session_id=session_id,
            cache_hit=ctx.cache_hit,
        )
    except Exception:
        logger.exception("Chat request failed | request_id=%s", request_id)
        raise
    finally:
        ctx.latency_ms["total"] = (time.perf_counter() - started) * 1000
        log_request_context(ctx)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}