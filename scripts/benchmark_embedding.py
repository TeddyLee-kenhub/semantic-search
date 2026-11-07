"""Compare full-precision BGE-M3 with the production int8 ONNX backend."""

import argparse
import gc
import json
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import httpx

from api.embedding import LocalBgeEmbedding
from api.service import Embedder
from api.settings import Settings
from api.store import MeiliDocumentStore


def document_texts(document: dict) -> list[str]:
    """Accept a single text or the API's multi-chunk document shape."""
    if "chunks" in document:
        return [chunk["text"] for chunk in document["chunks"]]
    return [document["text"]]


def load_dataset(path: Path) -> list[dict]:
    """Load query, positive document ID, and candidate documents from JSONL."""
    with path.open(encoding="utf-8") as source:
        records = [json.loads(line) for line in source if line.strip()]
    if not records:
        raise ValueError("Evaluation dataset is empty")
    for record in records:
        if (
            not isinstance(record.get("query"), str)
            or not isinstance(record.get("positive_id"), str)
            or not isinstance(record.get("documents"), list)
        ):
            raise ValueError("Invalid benchmark JSONL record")
        document_ids = {doc["id"] for doc in record["documents"]}
        if len(document_ids) != len(record["documents"]):
            raise ValueError("Candidate document IDs must be unique")
        for document in record["documents"]:
            texts = document_texts(document)
            if not texts or any(
                not isinstance(text, str) or not text.strip()
                for text in texts
            ):
                raise ValueError("Each candidate needs nonempty text chunks")
        if record["positive_id"] not in document_ids:
            raise ValueError("positive_id must identify a candidate document")
    return records


def prepare_records(
    records: list[dict], embedder: Embedder,
) -> list[dict]:
    """Apply production document splitting and strict query validation."""
    prepared = []
    for record in records:
        embedder.validate_text(record["query"])
        documents = [
            {
                **document,
                "chunks": [
                    {"text": text} for text in embedder.split_document(
                        document_texts(document)
                    )
                ],
            }
            for document in record["documents"]
        ]
        prepared.append({**record, "documents": documents})
    return prepared


def evaluate(
    encode: Callable[[list[str]], np.ndarray], records: list[dict],
    store: MeiliDocumentStore | None = None,
) -> dict[str, float]:
    """Rank best-chunk cosine or production Meilisearch candidates."""
    ranks = []
    for record in records:
        documents = record["documents"]
        query_vector = encode([record["query"]])[0]
        document_vectors = [
            encode(document_texts(document)) for document in documents
        ]
        if store is None:
            scores = [
                float(np.max(vectors @ query_vector))
                for vectors in document_vectors
            ]
            order = np.argsort(-np.asarray(scores), kind="stable")
            ordered_ids = [documents[index]["id"] for index in order]
        else:
            for document, vectors in zip(documents, document_vectors):
                store.replace_document(
                    "article", document["id"], "en", document["id"],
                    document_texts(document), vectors,
                )
            ordered_ids = [
                result.record_id for result in store.search(
                    query_vector, "en", ["article"],
                    len(documents), len(documents),
                )
            ]
            for document in documents:
                store.delete_document("article", document["id"])
        rank = ordered_ids.index(record["positive_id"]) + 1
        ranks.append(rank)
    count = len(records)
    return {
        "queries": count,
        "recall_at_1": sum(rank == 1 for rank in ranks) / count,
        "recall_at_3": sum(rank <= 3 for rank in ranks) / count,
        "mrr": float(np.mean([1.0 / rank for rank in ranks])),
        "mean_positive_rank": float(np.mean(ranks)),
    }


def delete_benchmark_index(settings: Settings) -> None:
    """Delete only this benchmark's unique index and await completion."""
    assert settings.meilisearch_index.startswith("semantic_search_benchmark_")
    with httpx.Client(
        base_url=settings.meilisearch_host, trust_env=False,
        timeout=settings.meilisearch_timeout,
        headers={"Authorization": "Bearer " + settings.meilisearch_api_key},
    ) as client:
        response = client.delete(f"/indexes/{settings.meilisearch_index}")
        if response.status_code == 404:
            return
        response.raise_for_status()
        task_id = response.json()["taskUid"]
        deadline = time.monotonic() + settings.meilisearch_task_timeout
        while time.monotonic() < deadline:
            response = client.get(f"/tasks/{task_id}")
            response.raise_for_status()
            status = response.json()["status"]
            if status == "succeeded":
                return
            if status not in {"enqueued", "processing"}:
                raise RuntimeError("Benchmark index deletion failed")
            time.sleep(0.1)
        raise TimeoutError("Benchmark index deletion timed out")


def main() -> None:
    """Run the full-precision and production int8 comparison offline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference-model",
        type=Path,
        default=Path("cache/models/bge-m3"),
    )
    parser.add_argument(
        "--int8-model",
        type=Path,
        default=Path("cache/models/bge-m3-onnx-int8"),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("tests/data/anatomy_retrieval.jsonl"),
    )
    parser.add_argument(
        "--meilisearch", action="store_true",
        help="Also evaluate production retrieval in a temporary remote index",
    )
    args = parser.parse_args()
    if args.meilisearch and not Settings.from_env().meilisearch_api_key:
        raise ValueError("--meilisearch requires MEILISEARCH_API_KEY")
    records = load_dataset(args.dataset)
    candidate_settings = replace(
        Settings.from_env(),
        model_path=args.int8_model,
    )
    candidate = LocalBgeEmbedding(candidate_settings)
    records = prepare_records(records, candidate)
    del candidate
    gc.collect()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # pylint: disable=import-outside-toplevel
    from sentence_transformers import SentenceTransformer

    started = time.monotonic()
    reference = SentenceTransformer(
        str(args.reference_model),
        device="cpu",
        local_files_only=True,
    )
    reference_metrics = evaluate(
        lambda texts: reference.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=1,
        ),
        records,
    )
    print(
        json.dumps(
            {
                "model": "BAAI/bge-m3-fp32",
                "seconds": round(time.monotonic() - started, 3),
                **reference_metrics,
            },
            ensure_ascii=False,
        )
    )
    del reference
    gc.collect()

    started = time.monotonic()
    candidate = LocalBgeEmbedding(candidate_settings)
    candidate_metrics = evaluate(candidate.embed, records)
    print(
        json.dumps(
            {
                "model": "onnx-community/bge-m3-ONNX-int8",
                "seconds": round(time.monotonic() - started, 3),
                **candidate_metrics,
            },
            ensure_ascii=False,
        )
    )
    if args.meilisearch:
        benchmark_settings = replace(
            candidate_settings,
            meilisearch_index="semantic_search_benchmark_" + uuid.uuid4().hex,
        )
        store = None
        try:
            store = MeiliDocumentStore(benchmark_settings)
            metrics = evaluate(candidate.embed, records, store)
            print(json.dumps(
                {"model": "int8-meilisearch", **metrics},
                ensure_ascii=False,
            ))
        finally:
            if store is not None:
                store.close()
            delete_benchmark_index(benchmark_settings)


if __name__ == "__main__":
    main()
