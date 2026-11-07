"""Shared fixtures using deterministic local embeddings."""

# Pytest fixtures are explicitly requested by test parameter name.
# pylint: disable=redefined-outer-name,unused-argument
import pytest
import httpx
from fastapi.testclient import TestClient

from api.settings import Settings
from tests.fakes import FakeMeilisearch, build_test_app


@pytest.fixture
def meili_server(monkeypatch):
    """Route only the storage client's requests to an in-memory server."""
    server = FakeMeilisearch()
    client_class = httpx.Client
    monkeypatch.setattr(
        "api.store.httpx.Client",
        lambda **kwargs: client_class(
            **kwargs, transport=httpx.MockTransport(server.handle)
        ),
    )
    return server


@pytest.fixture
def settings(tmp_path, meili_server):
    """Return isolated service paths and deterministic limits."""
    return Settings(
        model_path=tmp_path / "model",
        device="cpu",
        model_id="BAAI/bge-m3",
        embedding_dimension=1024,
        max_model_tokens=8192,
        max_request_bytes=8 * 1024 * 1024,
        max_concurrent_requests=2,
        max_search_candidates=100000,
    )


@pytest.fixture
def app(settings):
    """Build the app with mocked Meilisearch and fake embeddings."""
    return build_test_app(settings)


@pytest.fixture
def client(app):
    """Start app lifespan and provide an in-process HTTP client."""
    with TestClient(app) as test_client:
        yield test_client
