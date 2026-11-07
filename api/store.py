"""Meilisearch storage for locally generated, multi-chunk embeddings."""

import hashlib
import json
import re
import time
from collections.abc import Sequence
from urllib.parse import quote

import httpx
import numpy as np

from api.schemas import ContentType, Locale, SearchResult
from api.settings import Settings

PREPROCESSING_VERSION = "bge-m3-single-text-cls-normalized-v3"
SCHEMA_VERSION = "2"


class IndexCompatibilityError(RuntimeError):
    """Raised when an existing index does not match this service."""


class MeilisearchError(RuntimeError):
    """A sanitized Meilisearch transport, task, or response failure."""


class StoreUnavailableError(MeilisearchError):
    """Raised when the remote store is temporarily unreachable."""


class RequestRejectedError(MeilisearchError):
    """Raised when the remote server explicitly rejects a request."""


class MeiliDocumentStore:
    """Store one document with multiple externally computed chunk vectors."""

    def __init__(self, settings: Settings):
        """Connect, finish pending tasks, and verify the dedicated index."""
        if not re.fullmatch(r"[A-Za-z0-9_-]+", settings.meilisearch_index):
            raise ValueError("MEILISEARCH_INDEX must be an index identifier")
        identity = json.dumps(
            [settings.model_id, settings.embedding_dimension,
             PREPROCESSING_VERSION, SCHEMA_VERSION],
            separators=(",", ":"),
        )
        self.embedder_name = (
            "local_" + hashlib.sha256(identity.encode()).hexdigest()
        )
        self._index = settings.meilisearch_index
        self._path = f"/indexes/{self._index}"
        self._dimension = settings.embedding_dimension
        self._task_timeout = settings.meilisearch_task_timeout
        self._pending = False
        self._client = httpx.Client(
            base_url=settings.meilisearch_host,
            headers={
                "Authorization": f"Bearer {settings.meilisearch_api_key}",
            },
            timeout=settings.meilisearch_timeout,
            trust_env=False,
        )
        try:
            self._initialize()
        except BaseException:
            self._client.close()
            raise

    def _request(self, method: str, path: str, **kwargs) -> dict:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.HTTPError:
            raise StoreUnavailableError(
                "Meilisearch connection or request failed"
            ) from None
        if response.status_code in {502, 503, 504}:
            raise StoreUnavailableError(
                f"Meilisearch returned HTTP {response.status_code}"
            )
        if response.is_client_error and response.status_code != 408:
            raise RequestRejectedError(
                f"Meilisearch returned HTTP {response.status_code}"
            )
        if not response.is_success:
            raise MeilisearchError(
                f"Meilisearch returned HTTP {response.status_code}"
            )
        try:
            payload = response.json()
        except ValueError:
            raise MeilisearchError(
                "Meilisearch returned invalid JSON"
            ) from None
        if not isinstance(payload, dict):
            raise MeilisearchError("Meilisearch returned an invalid response")
        return payload

    def _wait_task(self, task_id: int) -> None:
        deadline = time.monotonic() + self._task_timeout
        while time.monotonic() < deadline:
            task = self._request("GET", f"/tasks/{task_id}")
            status = task.get("status")
            if status == "succeeded":
                self._pending = False
                return
            if status in {"failed", "canceled"}:
                self._pending = False
                # Upstream error messages can contain submitted text.
                raise MeilisearchError(
                    f"Meilisearch task {task_id} {status}"
                )
            if status not in {"enqueued", "processing"}:
                raise MeilisearchError("Meilisearch task has invalid status")
            time.sleep(0.1)
        raise MeilisearchError(
            f"Meilisearch task {task_id} did not finish before timeout"
        )

    def _mutate(self, method: str, path: str, **kwargs) -> None:
        self._pending = True
        try:
            task = self._request(method, path, **kwargs)
        except RequestRejectedError:
            self._pending = False
            raise
        task_id = task.get("taskUid")
        if not isinstance(task_id, int):
            raise MeilisearchError("Meilisearch did not return a task ID")
        self._wait_task(task_id)

    def _drain_tasks(self) -> None:
        deadline = time.monotonic() + self._task_timeout
        while time.monotonic() < deadline:
            tasks = self._request(
                "GET", "/tasks",
                params={
                    "indexUids": self._index,
                    "statuses": "enqueued,processing",
                    "limit": 1,
                },
            )
            if not isinstance(tasks.get("results"), list):
                raise MeilisearchError("Invalid pending task response")
            if not tasks["results"]:
                return
            time.sleep(0.1)
        raise MeilisearchError("Meilisearch index still has pending tasks")

    def _initialize(self) -> None:
        self._drain_tasks()
        # Listing by UID through GET keeps 404 handling narrowly scoped.
        try:
            response = self._client.get(self._path)
        except httpx.HTTPError:
            raise MeilisearchError("Meilisearch index lookup failed") from None
        if response.status_code == 404:
            self._mutate(
                "POST", "/indexes",
                json={"uid": self._index, "primaryKey": "id"},
            )
        elif not response.is_success:
            raise MeilisearchError(
                f"Meilisearch index lookup returned HTTP "
                f"{response.status_code}"
            )
        index = self._request("GET", self._path)
        if index.get("primaryKey") != "id":
            raise IndexCompatibilityError(
                "Meilisearch index primary key is incompatible"
            )
        settings = self._request("GET", self._path + "/settings")
        if not settings.get("embedders"):
            stats = self._request("GET", self._path + "/stats")
            if stats.get("numberOfDocuments") != 0:
                raise IndexCompatibilityError(
                    "Existing documents have no compatible model identity"
                )
            self._mutate(
                "PATCH", self._path + "/settings",
                json={
                    "embedders": self._expected_embedders(),
                    "searchableAttributes": [],
                    "filterableAttributes": ["locale", "type"],
                },
            )
        self._check_settings()

    def _expected_embedders(self) -> dict:
        return {
            self.embedder_name: {
                "source": "userProvided",
                "dimensions": self._dimension,
            },
        }

    def _check_settings(self) -> None:
        settings = self._request("GET", self._path + "/settings")
        if (
            settings.get("embedders") != self._expected_embedders()
            or settings.get("searchableAttributes") != []
            or settings.get("filterableAttributes") != ["locale", "type"]
        ):
            raise IndexCompatibilityError(
                "Meilisearch model/schema/settings are incompatible; "
                "use a new dedicated index and reindex source documents"
            )

    @staticmethod
    def _document_id(doc_type: ContentType, record_id: str) -> str:
        return hashlib.sha256(
            f"{doc_type}:{record_id}".encode()
        ).hexdigest()

    # Each argument is part of the public document replacement contract.
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def replace_document(
        self,
        doc_type: ContentType,
        record_id: str,
        locale: Locale,
        slug: str,
        chunks: Sequence[str],
        vectors: np.ndarray,
    ) -> int:
        """Atomically replace all chunks and wait for indexing to succeed."""
        if not chunks:
            self.delete_document(doc_type, record_id)
            return 0
        if (
            vectors.shape != (len(chunks), self._dimension)
            or not np.isfinite(vectors).all()
        ):
            raise ValueError("Invalid document vectors")
        document = {
            "id": self._document_id(doc_type, record_id),
            "type": doc_type,
            "record_id": record_id,
            "locale": locale,
            "slug": slug,
            "chunks": list(chunks),
            "_vectors": {
                self.embedder_name: {
                    "embeddings": vectors.tolist(),
                    "regenerate": False,
                },
            },
        }
        # POST replaces the whole document; PUT is a partial update in Meili.
        self._mutate(
            "POST", self._path + "/documents", json=[document]
        )
        return len(chunks)

    def delete_document(
        self, doc_type: ContentType, record_id: str
    ) -> None:
        """Idempotently delete one whole multi-vector document."""
        document_id = quote(self._document_id(doc_type, record_id), safe="")
        self._mutate(
            "DELETE", self._path + f"/documents/{document_id}"
        )

    # Match the public filtered search contract.
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def search(
        self,
        vector: np.ndarray,
        locale: Locale,
        types: Sequence[ContentType],
        limit: int,
        max_candidates: int,
    ) -> list[SearchResult]:
        """Find distinct documents and return best-chunk cosine similarity."""
        payload = self._request(
            "POST", self._path + "/search",
            json={
                "vector": vector.tolist(),
                "hybrid": {
                    "embedder": self.embedder_name,
                    "semanticRatio": 1,
                },
                "filter": [
                    f"locale = {json.dumps(locale)}",
                    [f"type = {json.dumps(value)}" for value in types],
                ],
                "limit": min(limit, max_candidates),
                "retrieveVectors": True,
                "attributesToRetrieve": [
                    "type", "record_id", "locale", "slug",
                ],
            },
        )
        hits = payload.get("hits")
        if not isinstance(hits, list):
            raise MeilisearchError("Meilisearch returned invalid search hits")
        results = [self._search_result(hit, vector) for hit in hits]
        return sorted(
            results, key=lambda result: result.score, reverse=True
        )[:limit]

    def _search_result(self, hit: dict, query: np.ndarray) -> SearchResult:
        try:
            embeddings = np.asarray(
                hit["_vectors"][self.embedder_name]["embeddings"],
                dtype=np.float32,
            )
        except (KeyError, TypeError, ValueError):
            raise MeilisearchError("Invalid stored vectors") from None
        if (
            embeddings.ndim != 2
            or embeddings.shape[0] == 0
            or embeddings.shape[1] != self._dimension
            or not np.isfinite(embeddings).all()
        ):
            raise MeilisearchError("Invalid stored vector shape or values")
        norms = np.linalg.norm(embeddings, axis=1)
        query_norm = np.linalg.norm(query)
        if np.any(norms == 0) or query_norm == 0:
            raise MeilisearchError("Cannot score zero-length vectors")
        score = float(
            np.max((embeddings @ query) / (norms * query_norm))
        )
        return SearchResult(
            type=hit["type"],
            record_id=hit["record_id"],
            locale=hit["locale"],
            slug=hit["slug"],
            score=max(-1.0, min(1.0, score)),
        )

    def has_pending_changes(self) -> bool:
        """Report an ambiguous write requiring service restart and recovery."""
        return self._pending

    def check_ready(self) -> bool:
        """Verify remote storage and model configuration are available."""
        if self._pending:
            return False
        self._check_settings()
        stats = self._request("GET", self._path + "/stats")
        return stats.get("isIndexing") is False

    def close(self) -> None:
        """Close pooled HTTP connections without deleting the remote index."""
        self._client.close()
