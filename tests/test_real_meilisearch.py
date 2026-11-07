"""Opt-in end-to-end tests using real local ONNX and Meilisearch."""

import os
import time
import uuid
from dataclasses import replace

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.embedding import LocalBgeEmbedding
from api.server import create_app
from api.settings import Settings
from tests.test_api import put_document


def delete_test_index(settings: Settings) -> None:
    """Remove only the randomly named index created by this test."""
    assert settings.meilisearch_index.startswith("semantic_search_test_")
    with httpx.Client(
        trust_env=False,
        timeout=settings.meilisearch_timeout,
        headers={"Authorization": "Bearer " + settings.meilisearch_api_key},
        base_url=settings.meilisearch_host,
    ) as client:
        response = client.delete(
            f"/indexes/{settings.meilisearch_index}"
        )
        if response.status_code == 404:
            return
        assert response.status_code == 202
        task_id = response.json()["taskUid"]
        deadline = time.monotonic() + settings.meilisearch_task_timeout
        while time.monotonic() < deadline:
            task = client.get(f"/tasks/{task_id}")
            assert task.status_code == 200
            status = task.json()["status"]
            if status == "succeeded":
                return
            assert status not in {"failed", "canceled"}
            time.sleep(0.1)
        raise TimeoutError("Test index deletion did not finish")


def verify_automatic_splitting(client, settings, embedder):
    """Check exact stored text, actual count, and atomic overflow rejection."""
    original = "股" * 6000
    expected_chunks = embedder.split_document([original])
    response = put_document(client, "long", [original])
    assert response.status_code == 200
    assert response.json()["chunks_count"] == len(expected_chunks)
    with httpx.Client(
        base_url=settings.meilisearch_host, trust_env=False,
        headers={
            "Authorization": "Bearer " + settings.meilisearch_api_key,
        },
    ) as remote:
        response = remote.get(
            f"/indexes/{settings.meilisearch_index}/documents"
        )
        response.raise_for_status()
        stored = response.json()["results"][0]
        assert stored["chunks"] == expected_chunks
        assert "".join(stored["chunks"]) == original
    response = put_document(client, "long", [original] * 256)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"
    results = client.post(
        "/v1/search", json={"q": "股骨", "locale": "zh"},
    ).json()["results"]
    assert results[0]["record_id"] == "long"
    response = client.post(
        "/v1/search", json={"q": "股" * 2046, "locale": "zh"},
    )
    assert response.status_code == 422
    assert client.delete(
        "/v1/documents/article/long"
    ).status_code == 204


@pytest.mark.skipif(
    os.getenv("RUN_REAL_MEILISEARCH_TESTS") != "1",
    reason="set RUN_REAL_MEILISEARCH_TESTS=1 to test the real search engine",
)
def test_real_model_and_meilisearch_lifecycle():
    """Verify filtering, multi-vectors, replacement, restart, and deletion."""
    settings = replace(
        Settings.from_env(),
        meilisearch_index="semantic_search_test_" + uuid.uuid4().hex,
    )
    embedder = LocalBgeEmbedding(settings)
    chunks = ["股骨是大腿骨。", "股骨连接髋关节和膝关节。"]
    document_vectors = embedder.embed(chunks)
    query_vector = embedder.embed([chunks[0]])[0]
    expected_score = float(np.max(document_vectors @ query_vector))
    assert expected_score > 0.95
    try:
        with TestClient(
            create_app(settings, embedder_factory=lambda _: embedder)
        ) as client:
            assert client.get("/health/ready").status_code == 200
            verify_automatic_splitting(client, settings, embedder)
            assert put_document(
                client, "femur", chunks
            ).status_code == 200
            assert put_document(
                client, "tibia", ["胫骨位于小腿。"]
            ).status_code == 200
            assert client.put(
                "/v1/documents/study_unit/english",
                json={
                    "locale": "en", "slug": "english",
                    "chunks": [{"text": "The femur is the thigh bone."}],
                },
            ).status_code == 200
            results = client.post(
                "/v1/search",
                json={"q": "股骨是大腿骨。", "locale": "zh"},
            ).json()["results"]
            assert results[0]["record_id"] == "femur"
            assert results[0]["score"] == pytest.approx(
                expected_score, abs=1e-5
            )
            assert len({result["record_id"] for result in results}) == 2
            assert all(result["locale"] == "zh" for result in results)
            assert client.post(
                "/v1/search",
                json={
                    "q": "股骨", "locale": "zh", "types": ["study_unit"],
                },
            ).json() == {"results": []}
            assert put_document(
                client, "femur", ["髌骨位于膝关节前方。"], slug="replaced"
            ).status_code == 200
            results = client.post(
                "/v1/search",
                json={"q": "髌骨位于膝关节前方。", "locale": "zh"},
            ).json()["results"]
            assert results[0]["slug"] == "replaced"
            assert results[0]["score"] == pytest.approx(1.0, abs=1e-5)

        with TestClient(
            create_app(settings, embedder_factory=lambda _: embedder)
        ) as client:
            assert client.get("/health/ready").status_code == 200
            assert client.post(
                "/v1/search",
                json={"q": "髌骨位于膝关节前方。", "locale": "zh"},
            ).json()["results"][0]["slug"] == "replaced"
            assert client.delete(
                "/v1/documents/article/femur"
            ).status_code == 204
            assert client.delete(
                "/v1/documents/article/femur"
            ).status_code == 204
            assert put_document(client, "tibia", []).status_code == 200
            assert client.post(
                "/v1/search", json={"q": "股骨", "locale": "zh"},
            ).json() == {"results": []}
    finally:
        delete_test_index(settings)
