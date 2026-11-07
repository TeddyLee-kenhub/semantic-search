# Kenhub Semantic Search

Internal, single-instance FastAPI service for semantic search over Rails
articles and study units. Rails owns authorization, content state, and
presentation. This service owns local embeddings and a dedicated Meilisearch
vector index.

## Layout

- `main.py`: starts one Uvicorn worker.
- `api/server.py`: FastAPI endpoints, body-size limit, and lifespan.
- `api/embedding.py`: offline local int8 ONNX BGE-M3 loading and normalized
  dense vectors.
- `api/store.py`: Meilisearch multi-vector documents, compatibility validation,
  synchronous task completion, and best-chunk cosine scoring.
- `api/service.py`: bounded async coordination around synchronous inference and
  storage.
- `api/schemas.py`: request and response contract.
- `openapi.yaml`: canonical internal HTTP contract.
- `Dockerfile`, `docker-compose.yml`: non-root image, Linux host networking,
  loopback-only API, and environment-based Meilisearch connection.

## Invariants

1. No application-level authentication is intentional. The service must stay
   on a private network and must not be published publicly.
2. Load only the pinned int8 ONNX model copied from the ignored local `cache/`
   into the image at `MODEL_PATH`; runtime inference is offline. Startup fails
   if the model is missing or an actual inference probe produces an invalid
   shape, dimension, non-finite vector or failed normalization.
3. Use BGE-M3 dense normalized embeddings for both indexing and querying.
   Pool the CLS token, matching the reference model, then L2-normalize.
   Never mix OpenAI vectors or vectors from a different model/preprocessing
   version.
4. Use only a dedicated Meilisearch index. Model revision, quantization,
   dimension, schema and preprocessing are encoded in the embedder identity.
   Never modify a populated incompatible index or adopt an existing Rails
   business index.
5. Compute every replacement embedding before submitting a whole-document
   replacement. Wait for Meilisearch tasks before acknowledging writes.
   Ambiguous writes disable readiness; restart waits for pending remote tasks.
   Hold the mutation lock across inference and commit for every PUT/DELETE.
   Explicit 4xx submission rejection except 408 is not an ambiguous write.
6. Run one process/container per dedicated index; no external writer is
   supported. One Meilisearch document contains all vectors for a record.
7. Keep model text and queries out of normal logs. Surface model/storage errors
   as errors, not empty search results.
8. Keep HTTP behavior aligned with `openapi.yaml`; any contract change must be
   coordinated with Rails.
9. Validate all texts, then infer/pool one text at a time under the model lock.
   Dynamic int8 quantization is batch-sensitive: changing batching requires a
   new preprocessing identity and a full reindex, not adoption of old vectors.
   Default `MAX_MODEL_TOKENS=2048` includes special tokens. Split oversized
   document chunks using the loaded tokenizer, preserve exact original text
   and order, and check every resulting substring before inference.
   More than 256 resulting chunks returns 422 without changing the document;
   `chunks_count` is the actual indexed count. Never split queries: overflow
   returns 422. The 8192 model ceiling OOMs under a 4 GiB limit: larger configured
   contexts require explicit memory provisioning and target-hardware tests.
10. Recheck readiness inside the storage lock, including queued operations.
    Finish remote operations even on repeated client cancellation. Transient
    read/probe outages return 503 and recover; incompatibility and ambiguous
    writes remain latched unavailable until restart/recovery.
    Inference waits on an async lock before starting a thread. Cancelled
    waiters submit no work; started work retains capacity until completion.
    Use pure ASGI admission middleware: HTTP cancellation must retain the
    request slot until downstream work finishes, not only until headers.
11. Enforce body size before FastAPI parses the body. Streamed overflow returns
    the same contract-shaped 413 as Content-Length overflow.
    Body parser HTTP 400 errors map to the contract's 422 invalid_request.
    Body reception has finite positive idle/total deadlines (10/30 seconds
    by default). Timeout returns 408 request_timeout, closes the connection
    and frees request capacity. Never apply upload deadlines to inference.
12. Reject non-positive request concurrency, candidate count and body-byte
    limits when loading settings, before startup can report readiness.

## Development

Requires Python 3.12 and uv, run commands from this directory:

```bash
uv sync
uv run --env-file .env python main.py
uv run pytest
uv run pylint api main.py tests scripts
docker compose config --quiet
```

Compose defaults to a four-CPU quota and a hard 4 GiB memory limit without
swap. Benchmark queries must be encoded separately from each document's
chunks, with single-text model batches and best-chunk scoring. The
benchmark applies the production document splitter to both model backends.
Queries remain strictly validated and are never split. The optional
`--meilisearch` benchmark uses/cleans a random dedicated test index.

Default inference is CPU-only ONNX Runtime; the production dependencies do
not include PyTorch or CUDA/NVIDIA packages. Download the pinned model into
`cache/models/bge-m3-onnx-int8` before building the image. Full-precision
reference weights for the optional quality benchmark belong in
`cache/models/bge-m3` and are excluded from the Docker build context. Both
directories are git-ignored. Meilisearch owns persistent data; Python needs no
data volume. Set `MEILISEARCH_HOST`, `MEILISEARCH_API_KEY`, and a dedicated
`MEILISEARCH_INDEX` via the environment. Never commit a key.

Search is pure vector (`semanticRatio: 1`), not hybrid. Score is recomputed as
maximum chunk cosine, not copied from Meilisearch's ranking score. Approximate
candidate retrieval can differ from the previous exact LanceDB scan.

The regular tests use deterministic fake embeddings. The opt-in real-model test
requires already-downloaded local weights:

```bash
RUN_REAL_MODEL_TESTS=1 MODEL_PATH=/path/to/bge-m3-onnx-int8 \
  uv run pytest tests/test_real_model.py
```

The real Meilisearch test creates/removes only a random test index:

```bash
RUN_REAL_MEILISEARCH_TESTS=1 uv run --env-file .env \
  pytest -q tests/test_real_meilisearch.py
```
