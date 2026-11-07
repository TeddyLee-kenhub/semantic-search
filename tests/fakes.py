"""Reusable local fakes for semantic-search tests."""

import hashlib
import json
from collections.abc import Sequence

import httpx
import numpy as np

from api.server import create_app
from api.settings import Settings
from api.store import MeiliDocumentStore


class FakeEmbedder:
    """Small deterministic embedder for API and storage tests."""

    # This fake intentionally implements only the Embedder protocol.
    # pylint: disable=too-few-public-methods
    def __init__(self, _settings: Settings):
        """Initialize a fake backend with no forced failure."""
        self.fail_on: str | None = None

    def validate_text(self, text: str) -> None:
        """Apply a simple upper bound to exercise the service path."""
        if len(text) > 16000:
            raise ValueError("text too long")

    def split_document(self, texts: Sequence[str]) -> list[str]:
        """Keep already-valid chunks unchanged for deterministic tests."""
        return list(texts)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return repeatable unit vectors for API tests."""
        if "__embedding_failure__" in texts:
            raise RuntimeError("forced embedding failure")
        if self.fail_on is not None and self.fail_on in texts:
            raise RuntimeError("forced embedding failure")
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            vector = np.zeros(1024, dtype=np.float32)
            vector[digest[0] % 1024] = 1.0
            vectors.append(vector)
        return np.stack(vectors) if vectors else np.empty((0, 1024))


def build_test_app(settings: Settings):
    """Create an isolated FastAPI app with deterministic test dependencies."""
    return create_app(
        settings=settings,
        embedder_factory=FakeEmbedder,
        store_factory=MeiliDocumentStore,
    )


class FakeMeilisearch:
    """In-memory HTTP fake exercising the production Meilisearch adapter."""

    # Only the HTTP transport entry point is needed by the tests.
    # pylint: disable=too-few-public-methods
    def __init__(self):
        self.indexes = {}
        self.tasks = {}
        self.requests = []
        self.fail_task = False
        self.hold_task = False
        self.unavailable = False

    def _task(self, action):
        task_id = len(self.tasks)
        if self.fail_task:
            status = "failed"
        elif self.hold_task:
            status = "enqueued"
        else:
            action()
            status = "succeeded"
        self.tasks[task_id] = {"uid": task_id, "status": status}
        return httpx.Response(202, json={"taskUid": task_id})

    def handle(self, request):
        """Implement the Meilisearch routes used by the storage adapter."""
        # This fake deliberately models the related routes in one place.
        # pylint: disable=too-many-return-statements,too-many-branches
        if self.unavailable:
            return httpx.Response(503, json={"message": "unavailable"})
        self.requests.append(request)
        path = request.url.path
        data = json.loads(request.content) if request.content else None
        if path == "/tasks":
            tasks = [
                task for task in self.tasks.values()
                if task["status"] in {"enqueued", "processing"}
            ]
            return httpx.Response(200, json={"results": tasks})
        if path.startswith("/tasks/"):
            return httpx.Response(
                200, json=self.tasks[int(path.split("/")[-1])]
            )
        if path == "/indexes" and request.method == "POST":
            return self._task(
                lambda: self.indexes.update({
                    data["uid"]: {
                        "primaryKey": data["primaryKey"],
                        "settings": {"embedders": {}},
                        "documents": {},
                    },
                })
            )
        parts = path.strip("/").split("/")
        index = self.indexes.get(parts[1])
        if index is None:
            return httpx.Response(404, json={"code": "index_not_found"})
        if len(parts) == 2:
            return httpx.Response(
                200, json={"primaryKey": index["primaryKey"]}
            )
        if parts[2] == "settings":
            if request.method == "PATCH":
                return self._task(
                    lambda: index["settings"].update(data)
                )
            return httpx.Response(200, json=index["settings"])
        if parts[2] == "stats":
            return httpx.Response(200, json={
                "numberOfDocuments": len(index["documents"]),
                "isIndexing": False,
            })
        if parts[2] == "documents":
            if request.method == "DELETE":
                return self._task(
                    lambda: index["documents"].pop(parts[3], None)
                )
            return self._task(
                lambda: index["documents"].update({
                    doc["id"]: doc for doc in data
                })
            )
        if parts[2] == "search":
            return self._search(index, data)
        raise AssertionError(f"Unexpected route {request.method} {path}")

    @staticmethod
    def _search(index, data):
        embedder = data["hybrid"]["embedder"]
        assert data["hybrid"]["semanticRatio"] == 1
        locale = json.loads(data["filter"][0].split(" = ")[1])
        types = [
            json.loads(value.split(" = ")[1])
            for value in data["filter"][1]
        ]
        query = np.asarray(data["vector"])
        hits = []
        for document in index["documents"].values():
            if document["locale"] != locale or document["type"] not in types:
                continue
            vectors = np.asarray(
                document["_vectors"][embedder]["embeddings"]
            )
            score = float(np.max(vectors @ query))
            hits.append((score, document))
        hits.sort(key=lambda item: item[0], reverse=True)
        return httpx.Response(
            200, json={"hits": [doc for _, doc in hits[:data["limit"]]]}
        )
