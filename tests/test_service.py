"""Concurrency and recovery tests for the search coordinator."""

import asyncio
import threading

import pytest

from api.service import SearchService
from api.store import MeilisearchError
from tests.fakes import FakeEmbedder


class BlockingStore:
    """Hold one ambiguous write while a second operation queues."""

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.pending = False
        self.next_called = False

    def fail_write(self):
        """Raise only after a second operation can wait on the lock."""
        self.started.set()
        assert self.release.wait(5)
        self.pending = True
        raise MeilisearchError("ambiguous write")

    def next_write(self):
        """Record an operation that must never run after ambiguity."""
        self.next_called = True

    def has_pending_changes(self):
        """Return the store's unresolved-write state."""
        return self.pending


@pytest.mark.parametrize("cancellations", [0, 1, 2])
def test_queued_operation_rechecks_readiness_inside_lock(
    settings, cancellations
):
    """A waiter cannot bypass the stop imposed by an ambiguous write."""
    async def scenario():
        store = BlockingStore()
        service = SearchService(settings, FakeEmbedder(settings), store)
        # pylint: disable=protected-access
        first = asyncio.create_task(
            service._run_store(store.fail_write)
        )
        assert await asyncio.to_thread(store.started.wait, 5)
        second = asyncio.create_task(
            service._run_store(store.next_write)
        )
        await asyncio.sleep(0.02)
        for _ in range(cancellations):
            first.cancel()
            await asyncio.sleep(0)
        assert not first.done()
        store.release.set()
        with pytest.raises(MeilisearchError, match="ambiguous"):
            await first
        with pytest.raises(RuntimeError, match="not ready"):
            await second
        assert not store.next_called
        assert not service.ready

    asyncio.run(scenario())


class BlockingEmbedder(FakeEmbedder):
    """Expose the lifetime of inference without running the real graph."""

    def __init__(self, settings):
        super().__init__(settings)
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = []

    def embed(self, texts):
        """Block the first invocation and record every submitted text."""
        self.calls.append(list(texts))
        self.started.set()
        assert self.release.wait(5)
        return super().embed(texts)


class RecordingStore:
    """Track document mutations without remote I/O."""

    def __init__(self):
        self.chunks = ["original"]
        self.operations = []

    def replace_document(self, *_args):
        """Commit a document after inference completes."""
        self.chunks = list(_args[-2])
        self.operations.append("put")
        return len(self.chunks)

    def delete_document(self, *_args):
        """Clear the document."""
        self.chunks = []
        self.operations.append("delete")

    def has_pending_changes(self):
        """There are no ambiguous operations in this fake."""
        return False

    def search(self, *_args):
        """Return an empty valid search response."""
        return []


def test_delete_waits_for_older_put_inference(settings):
    """A newer delete cannot be undone by an older in-flight replacement."""
    async def scenario():
        encoder = BlockingEmbedder(settings)
        store = RecordingStore()
        service = SearchService(settings, encoder, store)
        first = asyncio.create_task(service.replace_document(
            "article", "1", "zh", "one", ["new"]
        ))
        assert await asyncio.to_thread(encoder.started.wait, 5)
        second = asyncio.create_task(service.delete_document("article", "1"))
        await asyncio.sleep(0.02)
        assert not second.done()
        encoder.release.set()
        await asyncio.gather(first, second)
        assert store.operations == ["put", "delete"]
        assert not store.chunks

    asyncio.run(scenario())


@pytest.mark.parametrize("cancellations", [1, 2])
def test_cancelled_inference_holds_capacity_and_skips_waiters(
    settings, cancellations
):
    """Cancelled queued calls must never submit work to the thread pool."""
    async def scenario():
        encoder = BlockingEmbedder(settings)
        service = SearchService(settings, encoder, RecordingStore())
        active = asyncio.create_task(
            service.search("active", "en", ["article"], 10)
        )
        assert await asyncio.to_thread(encoder.started.wait, 5)
        waiting = asyncio.create_task(
            service.search("cancelled waiter", "en", ["article"], 10)
        )
        await asyncio.sleep(0.02)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        for _ in range(cancellations):
            active.cancel()
            await asyncio.sleep(0)
        assert not active.done()
        following = asyncio.create_task(
            service.search("following", "en", ["article"], 10)
        )
        await asyncio.sleep(0.02)
        assert encoder.calls == [["active"]]
        encoder.release.set()
        with pytest.raises(asyncio.CancelledError):
            await active
        assert await following == []
        assert encoder.calls == [["active"], ["following"]]

    asyncio.run(scenario())


def test_cancelled_put_never_commits_after_inference(settings):
    """Complete started inference but discard a cancelled replacement."""
    async def scenario():
        encoder = BlockingEmbedder(settings)
        store = RecordingStore()
        service = SearchService(settings, encoder, store)
        task = asyncio.create_task(service.replace_document(
            "article", "1", "en", "one", ["discarded"]
        ))
        assert await asyncio.to_thread(encoder.started.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        encoder.release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert store.chunks == ["original"]
        assert not store.operations

    asyncio.run(scenario())


@pytest.mark.parametrize("cancellations", [1, 2])
def test_cancelled_splitting_preserves_mutation_order(
    settings, monkeypatch, cancellations
):
    """Finish splitting on cancellation without encoding or committing."""
    async def scenario():
        encoder = BlockingEmbedder(settings)
        store = RecordingStore()
        service = SearchService(settings, encoder, store)

        def split(texts):
            encoder.started.set()
            assert encoder.release.wait(5)
            return list(texts)

        monkeypatch.setattr(encoder, "split_document", split)
        active = asyncio.create_task(service.replace_document(
            "article", "1", "en", "one", ["discarded"]
        ))
        assert await asyncio.to_thread(encoder.started.wait, 5)
        for _ in range(cancellations):
            active.cancel()
            await asyncio.sleep(0)
        following = asyncio.create_task(
            service.delete_document("article", "1")
        )
        await asyncio.sleep(0.02)
        assert not active.done()
        assert not following.done()
        assert store.chunks == ["original"]
        encoder.release.set()
        with pytest.raises(asyncio.CancelledError):
            await active
        await following
        assert not encoder.calls
        assert store.operations == ["delete"]

    asyncio.run(scenario())
