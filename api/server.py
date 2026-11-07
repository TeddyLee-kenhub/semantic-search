"""FastAPI routes and application lifecycle for semantic search."""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, Path, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from api.embedding import EmbeddingError, LocalBgeEmbedding
from api.schemas import (
    ContentType,
    ErrorResponse,
    RecordId,
    ReplaceDocumentRequest,
    ReplaceDocumentResponse,
    SearchRequest,
    SearchResponse,
)
from api.service import SearchService, ServiceUnavailableError
from api.settings import Settings
from api.store import MeiliDocumentStore, StoreUnavailableError

LOGGER = logging.getLogger("semantic_search.api")


class RequestConcurrencyMiddleware:
    """Keep request capacity until downstream work fully completes."""

    # ASGI middleware needs exactly one public entry point.
    # pylint: disable=too-few-public-methods
    def __init__(self, app: ASGIApp, max_requests: int):
        """Set the app and concurrent business-request capacity."""
        self._app = app
        self._slots = asyncio.Semaphore(max_requests)

    async def __call__(
        self, scope: Scope, receive: Receive, send: Send
    ) -> None:
        """Reject overload and propagate cancellation in the same app task."""
        if scope["type"] != "http" or not scope["path"].startswith("/v1/"):
            await self._app(scope, receive, send)
            return
        try:
            await asyncio.wait_for(self._slots.acquire(), timeout=0.01)
        except TimeoutError:
            response = JSONResponse(
                status_code=429,
                headers={"Retry-After": "1"},
                content={
                    "error": {
                        "code": "busy",
                        "message": "Search capacity is full; retry shortly.",
                    }
                },
            )
            await response(scope, receive, send)
            return
        started = time.monotonic()
        try:
            await self._app(scope, receive, send)
        finally:
            self._slots.release()
            LOGGER.debug(
                "Completed %s %s in %.3fs",
                scope["method"], scope["path"],
                time.monotonic() - started,
            )


class RequestBodyLimitMiddleware:
    """Reject oversized `/v1` request bodies, including chunked requests."""

    # ASGI middleware needs exactly one public entry point.
    # pylint: disable=too-few-public-methods
    def __init__(
        self, app: ASGIApp, max_bytes: int,
        idle_timeout: float = 10.0, body_timeout: float = 30.0,
    ):
        """Set the body size, idle and total upload limits."""
        self._app = app
        self._max_bytes = max_bytes
        self._idle_timeout = idle_timeout
        self._body_timeout = body_timeout

    async def __call__(
        self, scope: Scope, receive: Receive, send: Send
    ) -> None:
        """Count incoming request bytes and return a contract-shaped 413."""
        if scope["type"] != "http" or not scope["path"].startswith("/v1/"):
            await self._app(scope, receive, send)
            return

        content_length = next(
            (
                value
                for key, value in scope.get("headers", [])
                if key.lower() == b"content-length"
            ),
            None,
        )
        if content_length is not None and int(content_length) > self._max_bytes:
            await self._send_too_large(send)
            return

        body = bytearray()
        deadline = time.monotonic() + self._body_timeout
        while True:
            remaining = deadline - time.monotonic()
            try:
                if remaining <= 0:
                    raise TimeoutError
                message = await asyncio.wait_for(
                    receive(), timeout=min(self._idle_timeout, remaining)
                )
            except TimeoutError:
                LOGGER.warning(
                    "Request body read timed out on %s", scope["path"]
                )
                response = JSONResponse(
                    status_code=408,
                    headers={"Connection": "close"},
                    content={"error": {
                        "code": "request_timeout",
                        "message": "Request body was not received in time.",
                    }},
                )
                await response(scope, receive, send)
                return
            if message["type"] == "http.disconnect":
                return
            if message["type"] == "http.request":
                chunk = message.get("body", b"")
                if len(body) + len(chunk) > self._max_bytes:
                    await self._send_too_large(send)
                    return
                body.extend(chunk)
                if not message.get("more_body", False):
                    break

        buffered = bytes(body)
        del body
        delivered = False

        async def buffered_receive() -> Message:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": buffered}
            return await receive()

        await self._app(scope, buffered_receive, send)

    @staticmethod
    async def _send_too_large(send: Send) -> None:
        body = (
            b'{"error":{"code":"content_too_large",'
            b'"message":"Request body exceeds the configured size limit."}}'
        )
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


def create_app(
    settings: Settings | None = None,
    embedder_factory=None,
    store_factory=None,
) -> FastAPI:
    """Build the HTTP app, allowing lightweight dependencies in tests."""
    # App construction keeps routes and lifecycle bound to one settings object.
    # pylint: disable=too-many-locals
    app_settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Load the model and index before serving requests."""
        factory = embedder_factory or LocalBgeEmbedding
        store_builder = store_factory or MeiliDocumentStore
        embedder = await asyncio.to_thread(factory, app_settings)
        store = await asyncio.to_thread(
            store_builder,
            app_settings,
        )
        app.state.search_service = SearchService(
            app_settings, embedder, store
        )
        try:
            yield
        finally:
            await app.state.search_service.close()

    app = FastAPI(
        title="Kenhub Semantic Search API",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        RequestBodyLimitMiddleware,
        max_bytes=app_settings.max_request_bytes,
        idle_timeout=app_settings.request_body_idle_timeout,
        body_timeout=app_settings.request_body_timeout,
    )
    app.add_middleware(
        RequestConcurrencyMiddleware,
        max_requests=app_settings.max_concurrent_requests,
    )
    app.state.settings = app_settings

    @app.exception_handler(ServiceUnavailableError)
    @app.exception_handler(StoreUnavailableError)
    async def service_unavailable(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Report unavailable storage without exposing remote details."""
        LOGGER.error(
            "Search unavailable on %s %s: %s",
            request.method, request.url.path, exc,
        )
        return JSONResponse(
            status_code=503,
            content=ErrorResponse(
                error={
                    "code": "service_unavailable",
                    "message": "The model or vector store is not ready.",
                }
            ).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    @app.exception_handler(400)
    async def validation_error(
        _request: Request, _exc: Exception
    ) -> JSONResponse:
        """Keep validation errors in the stable public error envelope."""
        body = ErrorResponse(
            error={
                "code": "invalid_request",
                "message": "Request fields or values are invalid.",
            }
        )
        return JSONResponse(status_code=422, content=body.model_dump())

    @app.exception_handler(EmbeddingError)
    async def embedding_input_error(
        _request: Request, exc: EmbeddingError
    ) -> JSONResponse:
        """Report an explicit validation failure for model token limits."""
        body = ErrorResponse(
            error={"code": "invalid_request", "message": str(exc)}
        )
        return JSONResponse(status_code=422, content=body.model_dump())

    @app.exception_handler(Exception)
    async def unexpected_error(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Log an operation failure without returning private details."""
        # Storage/model backends can raise library-specific exceptions.
        # pylint: disable=broad-exception-caught
        LOGGER.exception(
            "Unhandled request failure on %s %s",
            request.method,
            request.url.path,
            exc_info=exc,
        )
        body = ErrorResponse(
            error={
                "code": "internal_error",
                "message": "The search operation failed.",
            }
        )
        return JSONResponse(status_code=500, content=body.model_dump())

    @app.exception_handler(404)
    async def route_not_found(
        _request: Request, _exc: Exception
    ) -> JSONResponse:
        """Return unknown routes using the service error envelope."""
        body = ErrorResponse(
            error={"code": "not_found", "message": "The route was not found."}
        )
        return JSONResponse(status_code=404, content=body.model_dump())

    @app.get("/health/live", tags=["Health"])
    async def liveness() -> dict[str, str]:
        """Return liveness without disclosing runtime configuration."""
        return {"status": "ok"}

    @app.get("/health/ready", tags=["Health"])
    async def readiness(request: Request) -> Response:
        """Require a loaded embedding model and accessible compatible store."""
        service: SearchService | None = getattr(
            request.app.state, "search_service", None
        )
        try:
            is_ready = service is not None and await service.check_ready()
        # Remote readiness failures must be logged and return 503.
        # pylint: disable=broad-exception-caught
        except Exception:
            LOGGER.exception("Readiness check failed")
            is_ready = False
        if not is_ready:
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "code": "service_unavailable",
                        "message": "The model or vector store is not ready.",
                    }
                },
            )
        return JSONResponse({"status": "ready"})

    @app.put(
        "/v1/documents/{type}/{record_id}",
        response_model=ReplaceDocumentResponse,
        tags=["Documents"],
    )
    async def replace_document(
        payload: ReplaceDocumentRequest,
        request: Request,
        doc_type: Annotated[
            ContentType, Path(alias="type")
        ],
        record_id: RecordId,
    ) -> ReplaceDocumentResponse:
        """Replace all indexed chunks for a document."""
        service: SearchService = request.app.state.search_service
        count = await service.replace_document(
            doc_type,
            record_id,
            payload.locale,
            payload.slug,
            [chunk.text for chunk in payload.chunks],
        )
        return ReplaceDocumentResponse(
            type=doc_type, record_id=record_id, chunks_count=count
        )

    @app.delete(
        "/v1/documents/{type}/{record_id}",
        status_code=204,
        tags=["Documents"],
    )
    async def delete_document(
        request: Request,
        doc_type: Annotated[
            ContentType, Path(alias="type")
        ],
        record_id: RecordId,
    ) -> Response:
        """Idempotently delete a document's vectors."""
        service: SearchService = request.app.state.search_service
        await service.delete_document(doc_type, record_id)
        return Response(status_code=204)

    @app.post(
        "/v1/search", response_model=SearchResponse, tags=["Search"]
    )
    async def search(
        payload: SearchRequest, request: Request
    ) -> SearchResponse:
        """Search by query text and aggregate chunk matches per document."""
        service: SearchService = request.app.state.search_service
        results = await service.search(
            payload.q, payload.locale, payload.types, payload.limit
        )
        return SearchResponse(results=results)

    return app


application = create_app()
