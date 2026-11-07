"""Meilisearch task completion, compatibility, and scoring tests."""

from dataclasses import replace

import numpy as np
import httpx
import pytest
from fastapi.testclient import TestClient

from api.settings import Settings
from api.store import (
    IndexCompatibilityError,
    MeiliDocumentStore,
    MeilisearchError,
    PREPROCESSING_VERSION,
)
from tests.fakes import build_test_app
from tests.test_api import put_document


def test_failed_task_preserves_old_document(client, settings, meili_server):
    """A terminal failed task is an error, not a successful replacement."""
    put_document(client, "1", ["old"])
    store = MeiliDocumentStore(settings)
    meili_server.fail_task = True
    with pytest.raises(MeilisearchError, match="failed"):
        store.replace_document(
            "article", "1", "zh", "new", ["new"],
            np.ones((1, 1024), dtype=np.float32),
        )
    assert not store.has_pending_changes()
    documents = meili_server.indexes[settings.meilisearch_index]["documents"]
    assert next(iter(documents.values()))["chunks"] == ["old"]
    store.close()


def test_pending_task_timeout_disables_service(settings, meili_server):
    """Do not return success or serve searches after an ambiguous write."""
    app = build_test_app(
        replace(settings, meilisearch_task_timeout=0.01)
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        meili_server.hold_task = True
        assert put_document(client, "1", ["pending"]).status_code == 500
        assert client.get("/health/ready").status_code == 503
        assert client.post(
            "/v1/search", json={"q": "pending", "locale": "zh"}
        ).status_code == 503


def test_restart_waits_for_pending_tasks(settings, meili_server):
    """A pending remote task must finish before a restarted service loads."""
    meili_server.tasks[0] = {"uid": 0, "status": "enqueued"}
    with pytest.raises(MeilisearchError, match="pending tasks"):
        MeiliDocumentStore(
            replace(settings, meilisearch_task_timeout=0.01)
        )
    meili_server.tasks[0]["status"] = "succeeded"
    store = MeiliDocumentStore(settings)
    assert store.check_ready()
    store.close()


def test_remote_failure_is_not_empty_search(app, meili_server):
    """Return an explicit failure when the remote search engine is down."""
    with TestClient(app, raise_server_exceptions=False) as client:
        meili_server.unavailable = True
        result = client.post(
            "/v1/search", json={"q": "anatomy", "locale": "zh"}
        )
        assert result.status_code == 503
        assert result.json()["error"]["code"] == "service_unavailable"


def test_transient_outage_recovers_without_restart(app, meili_server):
    """A failed health probe cannot permanently disable a healthy store."""
    with TestClient(app, raise_server_exceptions=False) as client:
        meili_server.unavailable = True
        assert client.get("/health/ready").status_code == 503
        meili_server.unavailable = False
        assert client.get("/health/ready").status_code == 200
        assert client.post(
            "/v1/search", json={"q": "anatomy", "locale": "zh"}
        ).status_code == 200


def test_incompatible_settings_stay_unavailable(
    app, settings, meili_server
):
    """Compatibility faults require explicit restart, unlike network faults."""
    with TestClient(app, raise_server_exceptions=False) as client:
        index_settings = meili_server.indexes[
            settings.meilisearch_index
        ]["settings"]
        original = index_settings["searchableAttributes"]
        index_settings["searchableAttributes"] = ["chunks"]
        assert client.get("/health/ready").status_code == 503
        index_settings["searchableAttributes"] = original
        assert client.get("/health/ready").status_code == 503
        assert client.post(
            "/v1/search", json={"q": "anatomy", "locale": "zh"}
        ).status_code == 503


def test_best_chunk_cosine_score_is_not_meili_ranking_score(
    settings, meili_server
):
    """Preserve cosine [-1, 1], including orthogonal and negative vectors."""
    store = MeiliDocumentStore(settings)
    query = np.zeros(1024, dtype=np.float32)
    query[0] = 1
    orthogonal = np.zeros(1024, dtype=np.float32)
    orthogonal[1] = 1
    store.replace_document(
        "article", "one", "en", "one", ["negative", "orthogonal"],
        np.stack([-query, orthogonal]),
    )
    store.replace_document(
        "article", "two", "en", "two", ["negative"],
        np.stack([-query]),
    )
    results = store.search(query, "en", ["article"], 10, 1000)
    assert [result.score for result in results] == pytest.approx([0, -1])
    assert meili_server.requests[-1].method == "POST"
    store.close()


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_timeout_is_rejected(settings, timeout):
    """Always bound remote HTTP and task waits."""
    with pytest.raises(ValueError, match="timeouts"):
        replace(settings, meilisearch_task_timeout=timeout)


@pytest.mark.parametrize("field", [
    "max_concurrent_requests", "max_search_candidates", "max_request_bytes",
])
@pytest.mark.parametrize("value", [0, -1])
def test_invalid_capacity_is_rejected(settings, field, value):
    """Reject unusable settings before startup can report readiness."""
    with pytest.raises(ValueError, match=field):
        replace(settings, **{field: value})


@pytest.mark.parametrize("name", [
    "MAX_CONCURRENT_REQUESTS", "MAX_SEARCH_CANDIDATES", "MAX_REQUEST_BYTES",
])
def test_invalid_environment_capacity_is_rejected(monkeypatch, name):
    """Reject invalid deployment configuration before creating the app."""
    monkeypatch.setenv(name, "0")
    with pytest.raises(ValueError, match=name.lower()):
        Settings.from_env()


def test_minimum_positive_capacities_are_valid(settings):
    """Keep the valid lower boundary usable for small deployments."""
    configured = replace(
        settings, max_concurrent_requests=1, max_search_candidates=1,
        max_request_bytes=1,
    )
    assert configured.max_concurrent_requests == 1


@pytest.mark.parametrize("name", [
    "request_body_idle_timeout", "request_body_timeout",
])
@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_invalid_upload_timeout_is_rejected(settings, name, value):
    """Do not allow disabled or unbounded upload timeout configuration."""
    with pytest.raises(ValueError, match=name):
        replace(settings, **{name: value})


def test_upload_timeouts_are_loaded_from_environment(monkeypatch):
    """Wire deployment limits into the request body middleware settings."""
    monkeypatch.setenv("REQUEST_BODY_IDLE_TIMEOUT", "2")
    monkeypatch.setenv("REQUEST_BODY_TIMEOUT", "5")
    configured = Settings.from_env()
    assert configured.request_body_idle_timeout == 2
    assert configured.request_body_timeout == 5


def test_model_key_is_hidden_from_settings_repr(settings):
    """Do not accidentally print a credential when logging settings."""
    assert "test-secret" not in repr(
        replace(settings, meilisearch_api_key="test-secret")
    )


@pytest.mark.parametrize("status", [400, 401, 403, 413, 429])
def test_rejected_write_does_not_disable_service(
    app, monkeypatch, status
):
    """A definite rejection does not leave an unresolved indexing task."""
    with TestClient(app, raise_server_exceptions=False) as client:
        # pylint: disable=protected-access
        store = app.state.search_service._store
        original_request = store._client.request

        def reject(method, path, **kwargs):
            if method == "POST" and path.endswith("/documents"):
                return httpx.Response(status, json={}, request=httpx.Request(
                    method, "http://meili" + path
                ))
            return original_request(method, path, **kwargs)

        monkeypatch.setattr(store._client, "request", reject)
        assert put_document(client, "1", ["valid"]).status_code == 500
        assert not store.has_pending_changes()
        assert client.get("/health/ready").status_code == 200
        assert client.post(
            "/v1/search", json={"q": "valid", "locale": "zh"}
        ).status_code == 200


@pytest.mark.parametrize("status", [408, 500, 502, 503, 504])
def test_uncertain_write_keeps_pending(settings, status, monkeypatch):
    """Timeout and server failures cannot prove the write was not accepted."""
    store = MeiliDocumentStore(settings)
    try:
        monkeypatch.setattr(
            store._client, "request",  # pylint: disable=protected-access
            lambda *a, **kw: httpx.Response(
                status, json={}, request=httpx.Request("POST", "http://meili")
            ),
        )
        with pytest.raises(MeilisearchError):
            store.replace_document(
                "article", "1", "zh", "one", ["valid"],
                np.ones((1, 1024), dtype=np.float32),
            )
        assert store.has_pending_changes()
    finally:
        store.close()


def test_cls_encoder_rejects_previous_mean_pooling_index(
    settings, monkeypatch, meili_server
):
    """Changing pooling cannot silently adopt an existing model's vectors."""
    monkeypatch.setattr(
        "api.store.PREPROCESSING_VERSION",
        "bge-m3-single-text-masked-mean-normalized-v2",
    )
    old_store = MeiliDocumentStore(settings)
    try:
        old_store.replace_document(
            "article", "1", "zh", "one", ["existing"],
            np.ones((1, 1024), dtype=np.float32),
        )
    finally:
        old_store.close()
    monkeypatch.setattr(
        "api.store.PREPROCESSING_VERSION", PREPROCESSING_VERSION
    )
    with pytest.raises(IndexCompatibilityError):
        MeiliDocumentStore(settings)
    documents = meili_server.indexes[settings.meilisearch_index]["documents"]
    assert next(iter(documents.values()))["chunks"] == ["existing"]
