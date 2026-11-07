"""Opt-in checks of the documented chunk limit on real Meilisearch."""

import os
import uuid
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from api.server import create_app
from api.settings import Settings
from tests.fakes import FakeEmbedder
from tests.test_api import put_document
from tests.test_real_meilisearch import delete_test_index


@pytest.mark.skipif(
    os.getenv("RUN_REAL_MEILISEARCH_TESTS") != "1",
    reason="set RUN_REAL_MEILISEARCH_TESTS=1 to test the real search engine",
)
def test_real_meilisearch_supports_256_vectors_per_document():
    """Store the API's maximum chunk count without duplicates or truncation."""
    settings = replace(
        Settings.from_env(),
        meilisearch_index="semantic_search_test_" + uuid.uuid4().hex,
    )
    try:
        with TestClient(
            create_app(settings, embedder_factory=FakeEmbedder)
        ) as client:
            chunks = [f"chunk {index}" for index in range(256)]
            response = put_document(client, "uuid.123~test", chunks)
            assert response.status_code == 200
            assert response.json()["chunks_count"] == 256
            results = client.post(
                "/v1/search",
                json={"q": chunks[-1], "locale": "zh", "limit": 10},
            ).json()["results"]
            assert len(results) == 1
            assert results[0]["record_id"] == "uuid.123~test"
            assert results[0]["score"] == pytest.approx(1.0)
    finally:
        delete_test_index(settings)
