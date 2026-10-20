# syntax=docker/dockerfile:1.7
# One image serves the API, the Streamlit UI and the one-shot ingest job;
# docker-compose.yml picks the command for each.

# ---- builder: dependencies + pre-downloaded models -------------------------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CPU-only torch is ~200 MB instead of >2 GB with CUDA; installed first so
# sentence-transformers does not pull the default CUDA build.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the two models into the image so containers start without a download.
# Keep in sync with app/utils/embeddings.py and RERANKER_MODEL in app/config.py.
RUN python -c "\
from sentence_transformers import CrossEncoder, SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# ---- runtime ---------------------------------------------------------------
FROM python:3.12-slim

ENV PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/opt/hf-cache \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CHROMA_DB_PATH=/data/chroma_db \
    RESOURCES_PATH=/app/resources

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /data \
    && chown app:app /data

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=app:app /opt/hf-cache /opt/hf-cache

WORKDIR /app
COPY --chown=app:app app ./app
COPY --chown=app:app resources ./resources
COPY --chown=app:app scripts ./scripts
COPY --chown=app:app streamlit_app.py ./

USER app
EXPOSE 8000 8501

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
