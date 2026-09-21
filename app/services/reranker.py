"""Cross-encoder reranking with a relevance threshold.

A cross-encoder reads the query and each chunk together, so it judges
relevance far more precisely than the bi-encoder used for dense retrieval.

Scores are always passed through a sigmoid so they fall in [0, 1] whatever
the model's own default is (cross-encoder/ms-marco-* models ship with an
Identity activation and return raw logits). RERANKER_SCORE_THRESHOLD is
therefore on the same 0-1 scale for any RERANKER_MODEL.
"""

import torch
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.RERANKER_MODEL
        logger.info("Loading cross-encoder reranker: %s", self.model_name)
        self._model: CrossEncoder = CrossEncoder(self.model_name, activation_fn=torch.nn.Sigmoid())

    def rerank(
        self,
        query: str,
        candidates: list[Document],
        top_k: int,
        threshold: float,
    ) -> tuple[list[Document], list[float]]:
        if not candidates:
            return [], []

        try:
            pairs = [(query, doc.page_content) for doc in candidates]
            raw_scores = self._model.predict(pairs)
        except Exception:
            # Fail closed: without relevance scores we cannot vouch for any chunk.
            logger.exception("Reranker prediction failed")
            return [], []

        scored = sorted(
            zip(candidates, (float(s) for s in raw_scores)),
            key=lambda pair: pair[1],
            reverse=True,
        )
        logger.debug("Reranker scores: %s", [round(s, 4) for _, s in scored])

        passing = [(doc, score) for doc, score in scored if score >= threshold]
        if not passing:
            logger.warning(
                "Reranker threshold %.3f filtered all %d candidates (best=%.4f)",
                threshold,
                len(candidates),
                scored[0][1],
            )
            return [], []

        docs: list[Document] = []
        scores: list[float] = []
        for doc, score in passing[:top_k]:
            doc.metadata["reranker_score"] = score
            docs.append(doc)
            scores.append(score)
        return docs, scores


# Module-level singleton — the model is loaded once, when this module is
# first imported (FastAPI startup via app.rag / app.main).
reranker = CrossEncoderReranker()
