"""Async service coordinating local inference and Meilisearch operations."""

import asyncio
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from api.schemas import ContentType, Locale, SearchResult
from api.settings import Settings
from api.store import IndexCompatibilityError, MeiliDocumentStore


class ServiceUnavailableError(RuntimeError):
    """Raised when an unsafe store state requires service recovery."""


class Embedder(Protocol):
    """Minimal interface required by the index and search service."""

    def validate_text(self, text: str) -> None:
        """Validate model-specific input constraints."""

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return normalized vectors for all input texts."""

    def split_document(self, texts: Sequence[str]) -> list[str]:
        """Return ordered input-preserving chunks within the token budget."""


class SearchService:
    """Keep model inference and remote-store operations off the event loop."""

    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        store: MeiliDocumentStore,
    ):
        """Bind runtime dependencies and synchronization primitives."""
        self._settings = settings
        self._embedder = embedder
        self._store = store
        self._store_lock = asyncio.Lock()
        self._mutation_lock = asyncio.Lock()
        self._inference_lock = asyncio.Lock()
        self.ready = True

    # These are the fields of the document replacement contract.
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    async def replace_document(
        self,
        doc_type: ContentType,
        record_id: str,
        locale: Locale,
        slug: str,
        chunks: Sequence[str],
    ) -> int:
        """Embed all chunks before changing the active indexed document."""
        self._require_ready()
        async with self._mutation_lock:
            chunks = await self._finish_thread(
                self._embedder.split_document, chunks
            )
            vectors = await self._run_inference(chunks)
            return await self._run_store(
                self._store.replace_document,
                doc_type,
                record_id,
                locale,
                slug,
                chunks,
                vectors,
            )

    async def delete_document(
        self, doc_type: ContentType, record_id: str
    ) -> None:
        """Delete a document under the same lock used by searches."""
        self._require_ready()
        async with self._mutation_lock:
            await self._run_store(
                self._store.delete_document, doc_type, record_id
            )

    async def search(
        self,
        query: str,
        locale: Locale,
        types: Sequence[ContentType],
        limit: int,
    ) -> list[SearchResult]:
        """Embed a query and search only matching locale/type chunks."""
        self._require_ready()
        vectors = await self._run_inference([query])
        results = await self._run_store(
            self._store.search,
            vectors[0],
            locale,
            types,
            limit,
            self._settings.max_search_candidates,
        )
        return results

    async def check_ready(self) -> bool:
        """Retry transient failures but latch incompatible index state."""
        if not self.ready:
            return False
        try:
            return await asyncio.to_thread(self._store.check_ready)
        except IndexCompatibilityError:
            self.ready = False
            raise

    async def close(self) -> None:
        """Close the remote store connection."""
        await asyncio.to_thread(self._store.close)

    async def _run_store(self, operation, *args):
        """Serialize reads/writes and finish background I/O on cancellation."""
        self._require_ready()
        async with self._store_lock:
            self._require_ready()
            try:
                return await self._finish_thread(operation, *args)
            except IndexCompatibilityError:
                self.ready = False
                raise
            finally:
                if self._store.has_pending_changes():
                    self.ready = False

    async def _run_inference(self, texts: Sequence[str]) -> np.ndarray:
        async with self._inference_lock:
            self._require_ready()
            vectors = await self._finish_thread(self._embedder.embed, texts)
            self._validate_vectors(vectors, len(texts))
            return vectors

    @staticmethod
    async def _finish_thread(operation, *args):
        """Keep capacity occupied until started work ends, even if cancelled."""
        task = asyncio.create_task(asyncio.to_thread(operation, *args))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
            task.result()
            raise

    def _require_ready(self) -> None:
        if not self.ready:
            raise ServiceUnavailableError("Search service is not ready")

    @staticmethod
    def _validate_vectors(vectors: np.ndarray, count: int) -> None:
        """Fail closed if the embedder violates the persisted vector schema."""
        if vectors.shape != (count, 1024):
            raise RuntimeError(
                f"Embedding returned shape {vectors.shape}; "
                f"expected ({count}, 1024)"
            )
        if not np.isfinite(vectors).all():
            raise RuntimeError("Embedding returned a non-finite vector")
