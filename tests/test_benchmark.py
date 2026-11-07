"""Regression tests for production-shaped embedding quality evaluation."""

import numpy as np

from scripts.benchmark_embedding import evaluate, prepare_records
from api.store import MeiliDocumentStore
from tests.fakes import FakeEmbedder


def test_preparation_splits_documents_but_not_queries(settings, monkeypatch):
    """Compare the same actual document chunks in both benchmark backends."""
    encoder = FakeEmbedder(settings)
    calls = []

    def split(texts):
        calls.append(texts)
        return ["one ", "two"]

    monkeypatch.setattr(encoder, "split_document", split)
    records = [{
        "query": "query", "positive_id": "a",
        "documents": [{"id": "a", "text": "one two"}],
    }]
    prepared = prepare_records(records, encoder)
    assert calls == [["one two"]]
    assert prepared[0]["query"] == "query"
    assert prepared[0]["documents"][0]["chunks"] == [
        {"text": "one "}, {"text": "two"},
    ]
    assert "chunks" not in records[0]["documents"][0]


def test_evaluation_separates_query_and_document_chunks():
    """Prevent joint int8 query/document encoding and use best chunk score."""
    calls = []
    vectors = {
        "query": [1.0, 0.0],
        "unrelated": [0.0, 1.0],
        "positive": [1.0, 0.0],
        "distractor": [0.8, 0.6],
    }

    def encode(texts):
        calls.append(texts)
        return np.asarray([vectors[text] for text in texts], dtype=np.float32)

    records = [{
        "query": "query", "positive_id": "a",
        "documents": [
            {"id": "a", "chunks": [
                {"text": "unrelated"}, {"text": "positive"},
            ]},
            {"id": "b", "text": "distractor"},
        ],
    }]
    metrics = evaluate(encode, records)
    assert calls == [
        ["query"], ["unrelated", "positive"], ["distractor"],
    ]
    assert metrics == {
        "queries": 1, "recall_at_1": 1.0, "recall_at_3": 1.0,
        "mrr": 1.0, "mean_positive_rank": 1.0,
    }


def test_evaluation_can_use_meilisearch_candidates(settings, meili_server):
    """Verify the optional remote path and isolate each query's candidates."""
    records = [{
        "query": "anatomy", "positive_id": "a",
        "documents": [
            {"id": "a", "chunks": [{"text": "anatomy"}]},
            {"id": "b", "text": "other"},
        ],
    }]
    encoder = FakeEmbedder(settings)
    store = MeiliDocumentStore(settings)
    try:
        metrics = evaluate(encoder.embed, records, store)
    finally:
        store.close()
    assert metrics["recall_at_1"] == 1.0
    assert not meili_server.indexes[
        settings.meilisearch_index
    ]["documents"]
