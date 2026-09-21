"""RBAC-aware semantic response cache on Redis Stack (RediSearch vectors).

Every entry is tagged with a namespace of role + sorted authorized
collections + RETRIEVAL_VERSION, and every lookup filters on that exact
namespace. A finance user and an HR user therefore never share an entry,
even for identical questions, and bumping RETRIEVAL_VERSION orphans all
old entries (they then expire via CACHE_TTL).

Redis being down, lacking the search module, or erroring at runtime only
disables caching — it never fails a chat request.
"""

import json
import threading
import time

import numpy as np
import redis
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.query.filter import Tag
from redisvl.schema import IndexSchema

from app.config import settings
from app.utils.embeddings import EMBEDDING_DIMS, get_embeddings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# How long to wait before retrying a Redis connection after a failure, so a
# Redis that starts after the API is picked up without per-request stalls.
_RECONNECT_INTERVAL_S = 30.0
_SOCKET_TIMEOUT_S = 2.0


class RBACSemanticCache:
    def __init__(self) -> None:
        self._client: redis.Redis | None = None
        self._index: SearchIndex | None = None
        self._available = False
        self._last_connect_attempt = 0.0
        self._connect_lock = threading.Lock()
        if settings.CACHE_ENABLED:
            self._try_connect()

    @property
    def available(self) -> bool:
        return self._available

    def _try_connect(self) -> None:
        self._last_connect_attempt = time.monotonic()
        try:
            self._client = redis.from_url(
                settings.REDIS_URL,
                socket_connect_timeout=_SOCKET_TIMEOUT_S,
                socket_timeout=_SOCKET_TIMEOUT_S,
            )
            self._client.ping()
            self._setup_index()
            self._available = True
            logger.info("Redis Stack cache connected successfully")
        except Exception as e:
            logger.warning(f"Redis unavailable, caching disabled: {e}")
            self._available = False

    def _ensure_available(self) -> bool:
        if not settings.CACHE_ENABLED:
            return False
        if self._available:
            return True
        if time.monotonic() - self._last_connect_attempt < _RECONNECT_INTERVAL_S:
            return False
        with self._connect_lock:
            if not self._available:
                self._try_connect()
        return self._available

    def _setup_index(self) -> None:
        # Index name includes the retrieval version for cache invalidation.
        index_name = f"rbac_cache_{settings.RETRIEVAL_VERSION}"
        schema = IndexSchema.from_dict(
            {
                "index": {"name": index_name, "prefix": index_name, "storage_type": "hash"},
                "fields": [
                    {"name": "namespace", "type": "tag"},
                    {"name": "question", "type": "text"},
                    {"name": "response_json", "type": "text"},
                    {
                        "name": "embedding",
                        "type": "vector",
                        "attrs": {
                            "dims": EMBEDDING_DIMS,
                            "distance_metric": "cosine",
                            "algorithm": "flat",
                            "datatype": "float32",
                        },
                    },
                ],
            }
        )
        self._index = SearchIndex(schema, redis_client=self._client)
        if not self._index.exists():
            self._index.create(overwrite=False)

    def _make_namespace(self, role: str, collections: list[str]) -> str:
        collections_str = "_".join(sorted(collections))
        return f"{role}:{collections_str}:{settings.RETRIEVAL_VERSION}"

    def _embed(self, text: str) -> list[float]:
        return get_embeddings().embed_query(text)

    def get(self, question: str, role: str, collections: list[str]) -> dict | None:
        if not self._ensure_available():
            return None
        try:
            namespace = self._make_namespace(role, collections)
            normalized = question.lower().strip()
            embedding = self._embed(normalized)

            query = VectorQuery(
                vector=embedding,
                vector_field_name="embedding",
                return_fields=["namespace", "response_json"],
                num_results=1,
                # Tag filter escapes ':' etc. in the namespace value.
                filter_expression=Tag("namespace") == namespace,
            )
            results = self._index.query(query)
            if not results:
                logger.debug("Cache MISS | role=%s | no entry in namespace", role)
                return None

            top = results[0]
            # Cosine distance is in [0, 2]; similarity = 1 - distance.
            distance = float(top.get("vector_distance", 1.0))
            similarity = 1.0 - distance

            # Defence in depth: never return an entry from another namespace.
            returned_ns = top.get("namespace")
            if isinstance(returned_ns, bytes):
                returned_ns = returned_ns.decode()
            if returned_ns != namespace:
                logger.warning("Cache namespace mismatch ignored | role=%s", role)
                return None

            if similarity >= settings.CACHE_SIMILARITY_THRESHOLD:
                logger.info(f"Cache HIT | role={role} | similarity={similarity:.3f}")
                return json.loads(top["response_json"])

            logger.debug("Cache MISS | role=%s | best similarity=%.3f", role, similarity)
            return None

        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
            return None

    def set(self, question: str, role: str, collections: list[str], response: dict) -> None:
        if not self._ensure_available():
            return
        try:
            namespace = self._make_namespace(role, collections)
            normalized = question.lower().strip()
            embedding = self._embed(normalized)

            keys = self._index.load(
                [
                    {
                        "namespace": namespace,
                        "question": normalized,
                        "response_json": json.dumps(response),
                        "embedding": np.array(embedding, dtype=np.float32).tobytes(),
                    }
                ]
            )
            # Expire the keys redisvl actually wrote.
            for key in keys:
                self._client.expire(key, settings.CACHE_TTL)

            logger.info(f"Cache SET | role={role} | namespace={namespace}")
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")


# Module-level singleton
semantic_cache = RBACSemanticCache()
