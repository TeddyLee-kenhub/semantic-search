"""HTTP and persistence tests for the semantic-search API."""

import json
import asyncio
import threading
from dataclasses import replace
import pytest
import httpx
from fastapi.testclient import TestClient

from api.embedding import EmbeddingError
from api.store import MeiliDocumentStore
from api.server import (
    RequestBodyLimitMiddleware, RequestConcurrencyMiddleware, create_app,
)
from tests.fakes import build_test_app
from tests.test_service import BlockingEmbedder


def put_document(client, record_id, text, *, locale="zh", slug=None):
    """Create or replace one article using the public API."""
    return client.put(
        f"/v1/documents/article/{record_id}",
        json={
            "locale": locale,
            "slug": slug or f"article-{record_id}",
            "chunks": [{"text": value} for value in text],
        },
    )


def test_health_checks(client):
    """Expose liveness and readiness without authentication."""
    assert client.get("/health/live").json() == {"status": "ok"}
    assert client.get("/health/ready").json() == {"status": "ready"}


@pytest.mark.parametrize("cancellations", [1, 2])
@pytest.mark.parametrize("phase", ["inference", "store"])
def test_http_cancelled_inference_retains_admission_capacity(
    settings, cancellations, phase, monkeypatch
):
    """Cancellation must retain slots until inference or remote work ends."""
    async def scenario():
        encoder = BlockingEmbedder(settings)
        application = create_app(
            replace(settings, max_concurrent_requests=1),
            embedder_factory=lambda _: encoder,
        )
        async with application.router.lifespan_context(application):
            started = encoder.started
            release = encoder.release
            if phase == "store":
                encoder.release.set()
                started = threading.Event()
                release = threading.Event()
                # pylint: disable=protected-access
                store = application.state.search_service._store
                original = store.search

                def slow_search(*args):
                    started.set()
                    assert release.wait(5)
                    return original(*args)

                monkeypatch.setattr(store, "search", slow_search)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=application),
                base_url="http://test",
            ) as http:
                active = asyncio.create_task(http.post(
                    "/v1/search", json={"q": "active", "locale": "en"}
                ))
                assert await asyncio.to_thread(started.wait, 5)
                try:
                    for _ in range(cancellations):
                        active.cancel()
                        await asyncio.sleep(0)
                    assert not active.done()
                    assert (await http.get("/health/live")).status_code == 200
                    response = await asyncio.wait_for(http.post(
                        "/v1/search", json={"q": "waiting", "locale": "en"}
                    ), timeout=1)
                    assert response.status_code == 429
                    assert response.headers["Retry-After"] == "1"
                    assert response.json()["error"]["code"] == "busy"
                    assert encoder.calls == [["active"]]
                finally:
                    release.set()
                    with pytest.raises(asyncio.CancelledError):
                        await active
                following = await http.post(
                    "/v1/search", json={"q": "following", "locale": "en"}
                )
                assert following.status_code == 200
                assert encoder.calls == [["active"], ["following"]]

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["size", "validation", "backend"])
def test_request_capacity_is_released_after_http_errors(settings, failure):
    """Early rejections and unhandled failures cannot leak request capacity."""
    application = build_test_app(replace(
        settings, max_concurrent_requests=1, max_request_bytes=256
    ))
    with TestClient(application, raise_server_exceptions=False) as http:
        for _ in range(3):
            if failure == "size":
                response = http.post("/v1/search", content=b"x" * 257)
                assert response.status_code == 413
            elif failure == "validation":
                response = http.post(
                    "/v1/search", content=b"\xff",
                    headers={"content-type": "application/json"},
                )
                assert response.status_code == 422
            else:
                response = http.post("/v1/search", json={
                    "q": "__embedding_failure__", "locale": "en",
                })
                assert response.status_code == 500
        assert http.post(
            "/v1/search", json={"q": "healthy", "locale": "en"}
        ).status_code == 200


def test_document_write_search_and_delete(client):
    """Write, find, and idempotently remove one document."""
    written = put_document(client, "123", ["skull structure", "cranial bones"])
    assert written.status_code == 200
    assert written.json() == {
        "type": "article",
        "record_id": "123",
        "chunks_count": 2,
    }

    result = client.post(
        "/v1/search",
        json={
            "q": "skull structure",
            "locale": "zh",
            "types": ["article"],
            "limit": 10,
        },
    )
    assert result.status_code == 200
    assert len(result.json()["results"]) == 1
    assert result.json()["results"][0]["record_id"] == "123"
    assert result.json()["results"][0]["score"] == pytest.approx(1.0)

    deleted = client.delete("/v1/documents/article/123")
    assert deleted.status_code == 204
    assert client.delete("/v1/documents/article/123").status_code == 204
    assert client.post(
        "/v1/search",
        json={"q": "skull structure", "locale": "zh"},
    ).json() == {"results": []}


def test_split_document_count_and_atomic_failure(app, monkeypatch):
    """Return actual count and retain old content if splitting fails."""
    with TestClient(app) as http:
        encoder = app.state.search_service._embedder  # pylint: disable=protected-access

        def split(texts):
            if texts == ["overflow"]:
                raise EmbeddingError(
                    "Document exceeds 256 chunks after automatic splitting"
                )
            if texts == ["one two"]:
                return ["one ", "two"]
            return list(texts)

        monkeypatch.setattr(encoder, "split_document", split)
        response = put_document(http, "split", ["one two"])
        assert response.status_code == 200
        assert response.json()["chunks_count"] == 2
        response = put_document(http, "split", ["overflow"])
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_request"
        results = http.post(
            "/v1/search", json={"q": "two", "locale": "zh"}
        ).json()["results"]
        assert results[0]["record_id"] == "split"
        assert results[0]["score"] == pytest.approx(1.0)


def test_empty_chunks_clear_document_index(client):
    """Treat an empty replacement as deleting all document chunks."""
    put_document(client, "10", ["old content"])
    response = put_document(client, "10", [])
    assert response.status_code == 200
    assert response.json()["chunks_count"] == 0
    assert client.post(
        "/v1/search", json={"q": "old content", "locale": "zh"}
    ).json() == {"results": []}


def test_replacement_is_idempotent_and_removes_old_chunks(
    client, settings, meili_server
):
    """Replace old chunk rows and avoid duplicates on repeated requests."""
    put_document(client, "7", ["old phrase", "other old phrase"])
    put_document(client, "7", ["new phrase"])
    put_document(client, "7", ["new phrase"])

    rows = list(
        meili_server.indexes[settings.meilisearch_index]["documents"].values()
    )
    assert len(rows) == 1
    assert [row["chunks"] for row in rows] == [["new phrase"]]


def test_locale_and_type_filters_apply_in_vector_search(client):
    """Apply locale and type filters before vector candidates are selected."""
    put_document(client, "1", ["same phrase"], locale="zh")
    client.put(
        "/v1/documents/study_unit/2",
        json={
            "locale": "en",
            "slug": "same",
            "chunks": [{"text": "same phrase"}],
        },
    )

    zh_results = client.post(
        "/v1/search",
        json={
            "q": "same phrase",
            "locale": "zh",
            "types": ["study_unit"],
        },
    )
    en_results = client.post(
        "/v1/search", json={"q": "same phrase", "locale": "en"}
    )
    assert zh_results.json() == {"results": []}
    assert en_results.json()["results"][0]["record_id"] == "2"


@pytest.mark.parametrize(
    "payload",
    [
        {"q": "  ", "locale": "zh"},
        {"q": "query", "locale": "de"},
        {"q": "query", "locale": "zh", "types": []},
        {"q": "query", "locale": "zh", "limit": 0},
    ],
)
def test_invalid_search_payload_returns_contract_error(client, payload):
    """Reject unsupported locale, blank text, and invalid limits."""
    response = client.post("/v1/search", json=payload)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


def test_invalid_document_path_and_payload(client):
    """Reject document types outside the supported enum."""
    response = client.put(
        "/v1/documents/video/1",
        json={"locale": "zh", "slug": "x", "chunks": []},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


def test_request_body_limit(client):
    """Reject a request larger than the documented body limit."""
    response = client.put(
        "/v1/documents/article/1",
        content=b"x" * (8 * 1024 * 1024 + 1),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "content_too_large"


def test_streaming_body_limit_uses_contract_413(settings):
    """Count chunked bodies before the framework can swallow the error."""
    app = build_test_app(replace(settings, max_request_bytes=64))
    with TestClient(app) as client:
        def chunks():
            yield b'{"q":"' + b"x" * 40
            yield b"x" * 40 + b'","locale":"zh"}'

        response = client.post(
            "/v1/search", content=chunks(),
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "content_too_large"


def test_streaming_body_within_limit_is_parsed(client):
    """Replay buffered streaming bodies without changing normal parsing."""
    def chunks():
        yield b'{"q":"anatomy",'
        yield b'"locale":"zh"}'

    response = client.post(
        "/v1/search", content=chunks(),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 200
    assert response.json() == {"results": []}


@pytest.mark.parametrize("size", [63, 64, 65])
def test_fragmented_asgi_body_limit_boundary(size):
    """Enforce exact byte boundaries across separate ASGI receive events."""
    async def scenario():
        messages = iter([
            {"type": "http.request", "body": b"x" * 32, "more_body": True},
            {"type": "http.request", "body": b"x" * (size - 32)},
        ])
        responses = []
        called = False

        async def receive():
            return next(messages)

        async def send(message):
            responses.append(message)

        async def downstream(_scope, read, _send):
            nonlocal called
            called = True
            assert (await read())["body"] == b"x" * size

        middleware = RequestBodyLimitMiddleware(downstream, max_bytes=64)
        await middleware(
            {"type": "http", "path": "/v1/search"}, receive, send
        )
        assert called is (size <= 64)
        if size > 64:
            assert responses[0]["status"] == 413
            assert json.loads(responses[1]["body"])["error"]["code"] == (
                "content_too_large"
            )

    asyncio.run(scenario())


def test_disconnected_body_does_not_reach_application():
    """Stop buffering when the client disconnects before body completion."""
    async def scenario():
        messages = iter([
            {"type": "http.request", "body": b"{", "more_body": True},
            {"type": "http.disconnect"},
        ])

        async def receive():
            return next(messages)

        async def forbidden(*_args):
            pytest.fail("A disconnected request must not be processed")

        await RequestBodyLimitMiddleware(forbidden, 64)(
            {"type": "http", "path": "/v1/search"}, receive, forbidden
        )

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["idle", "total", "complete", "cancel"])
def test_upload_deadlines_and_capacity_recovery(mode):
    """Bound incomplete uploads without timing out downstream processing."""
    async def scenario():
        received = 0
        called = False
        responses = []

        async def receive():
            nonlocal received
            received += 1
            if mode in {"idle", "cancel"} and received > 1:
                await asyncio.sleep(5)
            elif mode == "total":
                await asyncio.sleep(0.015)
            return {
                "type": "http.request", "body": b"x",
                "more_body": mode != "complete",
            }

        async def send(message):
            responses.append(message)

        async def downstream(_scope, read, _send):
            nonlocal called
            assert (await read())["body"] == b"x"
            await asyncio.sleep(0.07)
            called = True

        idle = 0.03 if mode != "total" else 0.1
        middleware = RequestConcurrencyMiddleware(
            RequestBodyLimitMiddleware(
                downstream, 64, idle_timeout=idle, body_timeout=0.06
            ), max_requests=1,
        )
        scope = {"type": "http", "path": "/v1/search", "method": "POST"}
        task = asyncio.create_task(middleware(scope, receive, send))
        if mode == "cancel":
            await asyncio.sleep(0.01)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not responses
        else:
            await asyncio.wait_for(task, timeout=1)
            if mode == "complete":
                assert called
                assert not responses
            else:
                assert not called
                assert responses[0]["status"] == 408
                assert (b"connection", b"close") in responses[0]["headers"]
                assert json.loads(responses[1]["body"]) == {"error": {
                    "code": "request_timeout",
                    "message": "Request body was not received in time.",
                }}

        async def complete_receive():
            return {"type": "http.request", "body": b"x"}

        responses.clear()
        await middleware(scope, complete_receive, send)
        assert called
        assert not responses

    asyncio.run(scenario())


def test_invalid_json_does_not_return_framework_default(client):
    """Use the service error schema for malformed JSON."""
    response = client.post(
        "/v1/search", content="{", headers={"content-type": "application/json"}
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


def test_invalid_utf8_json_uses_contract_error(client):
    """Parser HTTP errors share the validation status and safe envelope."""
    response = client.post(
        "/v1/search", content=b"\xff",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {
        "code": "invalid_request",
        "message": "Request fields or values are invalid.",
    }}


def test_embedding_failure_preserves_previous_index(
    app, settings, meili_server
):
    """Keep the old searchable rows if new document embedding fails."""
    with TestClient(app, raise_server_exceptions=False) as test_client:
        original = put_document(test_client, "22", ["existing content"])
        failed = put_document(
            test_client, "22", ["__embedding_failure__"]
        )
        assert original.status_code == 200
        assert failed.status_code == 500

    rows = list(
        meili_server.indexes[settings.meilisearch_index]["documents"].values()
    )
    assert [row["chunks"] for row in rows] == [["existing content"]]


def test_vectors_survive_service_restart(settings):
    """Keep persisted chunk vectors available after recreating the app."""
    first_app = create_test_app(settings)
    with TestClient(first_app) as first_client:
        response = put_document(first_client, "44", ["persisted content"])
        assert response.status_code == 200

    restarted_app = create_test_app(settings)
    with TestClient(restarted_app) as restarted_client:
        response = restarted_client.post(
            "/v1/search",
            json={"q": "persisted content", "locale": "zh"},
        )
        assert response.status_code == 200
        assert response.json()["results"][0]["record_id"] == "44"


def create_test_app(settings):
    """Create a fresh app using the same remote index and fake embeddings."""
    return build_test_app(settings)


def test_model_metadata_mismatch_fails_loudly(settings):
    """Reject a persisted index created for another model identifier."""
    MeiliDocumentStore(settings).close()
    with pytest.raises(RuntimeError, match="incompatible"):
        MeiliDocumentStore(replace(settings, model_id="different-model"))


def test_unidentified_existing_documents_fail_loudly(settings, meili_server):
    """Refuse to adopt or modify an existing populated business index."""
    meili_server.indexes[settings.meilisearch_index] = {
        "primaryKey": "id",
        "settings": {"embedders": {}},
        "documents": {"existing": {"id": "existing"}},
    }
    with pytest.raises(RuntimeError, match="no compatible model"):
        MeiliDocumentStore(settings)


def test_openapi_spec_is_available(client):
    """Publish the routes without application-level auth requirements."""
    spec = client.get("/openapi.json").json()
    assert "/v1/documents/{type}/{record_id}" in spec["paths"]
    assert "/v1/search" in spec["paths"]
    assert "securitySchemes" not in spec.get("components", {})


def test_error_responses_are_json(client):
    """Return a consistent error envelope for missing required fields."""
    response = client.post("/v1/search", content=json.dumps({"q": "x"}))
    assert response.status_code == 422
    assert set(response.json()) == {"error"}
