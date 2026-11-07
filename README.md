# Kenhub Semantic Search

Internal, single-instance FastAPI service for semantic search over Kenhub
articles and study units. ONNX Runtime loads a local int8 BGE-M3 model;
Meilisearch stores and retrieves the externally generated dense vectors.
The service does not call Rails, PostgreSQL, Redis, OpenAI, or any model
endpoint. Meilisearch does not load or download an embedding model.

The service has no application-level authentication. Keep it on a trusted
internal network and do not publish its port to the Internet. Rails remains
responsible for caller authorization and for resolving current published
records from search result IDs.

## API

The canonical interface is [openapi.yaml](./openapi.yaml). Interactive docs
are available at `/docs` when the service is running.

- `PUT /v1/documents/{type}/{record_id}` replaces every chunk for one record.
- `DELETE /v1/documents/{type}/{record_id}` idempotently clears its chunks.
- `POST /v1/search` returns distinct records ordered by best-chunk cosine
  similarity (higher is better).
- `GET /health/live` and `GET /health/ready` provide liveness/readiness.

The OpenAPI contract sets an 8 MiB request-body cap, up to 256 chunks per
document and up to 16,000 Unicode characters per chunk. The default inference
limit is **2048 tokens including special tokens**, below BGE-M3's 8192-token
ceiling. The Python service uses its loaded tokenizer to recursively split
oversized document chunks, preferring nearby paragraph/sentence boundaries.
Every resulting substring is re-tokenized and checked including special
tokens; concatenating the pieces of each input reproduces its exact original
text. No decoding, normalization, overlap, or truncation is applied to stored
text. Already-valid chunks are unchanged; Rails needs no tokenizer.
`chunks_count` is the actual indexed count, which may exceed the submitted
count. More than 256 resulting chunks returns 422 before inference or any
write; the old document remains searchable. Queries are never split and
token overflow still returns 422. Search has
no fixed relevance threshold; unrelated queries can return nearest neighbors.

## Local setup

Requires Python 3.12 and uv:

```bash
uv sync
cp .env.example .env
# Set the API key in .env via your secure configuration mechanism.
uv run --env-file .env python main.py
uv run pytest
```

The supplied local connection was tested against Meilisearch 1.16.0.
Use a dedicated `MEILISEARCH_INDEX`, not an existing Rails search index.
The API key must allow index/settings/document/task access for that index.
Credentials are read from the environment and are never committed.

The production runtime uses CPU-only ONNX Runtime and does not install
PyTorch, CUDA, or Sentence Transformers. The optional `eval` dependency group
installs the full-precision PyTorch model for local quality comparisons.

## Model preparation and offline deployment

The production model is
[`onnx-community/bge-m3-ONNX`](https://huggingface.co/onnx-community/bge-m3-ONNX)
at revision `25b9af8e87a38eb120cfe87125383677b9cd309e`. Its model card declares
MIT and identifies `BAAI/bge-m3` as the base model. Download the pinned ONNX
graph and tokenizer on a connected machine:

```bash
uvx hf download onnx-community/bge-m3-ONNX \
  config.json tokenizer.json tokenizer_config.json \
  special_tokens_map.json onnx/model_int8.onnx \
  --revision 25b9af8e87a38eb120cfe87125383677b9cd309e \
  --local-dir cache/models/bge-m3-onnx-int8
```

The Git-ignored `cache/` directory retains downloaded weights locally while
keeping only `.gitkeep` in source control. Docker copies the 568 MB int8 model
into `/models/bge-m3-int8` at build time; it is part of the image, so hosts do
not need a model bind mount. The runtime is offline and never downloads a
model during startup or a request. `EMBEDDING_MODEL_ID` includes the pinned
revision and `int8` quantization in persisted compatibility metadata.

Build the image after downloading the model:

```bash
docker compose build
```

On the deployment host, start the self-contained image:

```bash
docker compose up -d
docker compose ps
docker compose logs semantic-search
```

The default Compose setup uses Linux host networking so the container can
reach Meilisearch at the supplied `127.0.0.1:7700`. The API binds to
`127.0.0.1:8850`, not a public interface. A container's own loopback address
would not reach host Meilisearch with ordinary bridge networking.
Rails on the host can call `http://127.0.0.1:8850`; clients on another host or
Docker network need an explicitly configured private interface or trusted
reverse proxy. Set `APP_HOST` to that private address only after restricting
network access. Do not expose this unauthenticated service publicly.
With Docker Desktop or a remote daemon, host networking refers to the
daemon's host/VM, not necessarily the shell's host. In this environment,
container-local API checks passed but shell access to port 8850 timed out.
Validate client connectivity in the target deployment; use the daemon's
private network or a trusted proxy if those hosts differ.

The container runs one Uvicorn worker as a non-root user. Model weights are
read-only image files. The Python service no longer needs a data volume.
Readiness requires model loading, compatible remote settings, and an
accessible Meilisearch index with no unresolved service write.
Startup runs an actual inference probe and checks output shape, finite values
and normalization before initializing the index or accepting traffic.

## Storage, compatibility, and backup

Each article or study unit is one Meilisearch document with a deterministic
SHA-256 ID derived from `(type, record_id)`, its text chunks, and one vector
per chunk. The embedder is `userProvided`, with 1024 dimensions and a name
hashed from model revision, quantization, schema, and preprocessing identity.
An incompatible existing index fails startup instead of being modified or
reindexed silently. Empty unconfigured dedicated indexes can be initialized.
The encoder uses BGE-M3's CLS token output followed by L2 normalization,
with one text per inference batch. This matches the reference model's pooling
instead of the previous masked mean pooling. The preprocessing identity is
`bge-m3-single-text-cls-normalized-v3`.
Existing indexes from either mean-pooling implementation must be replaced
with a new dedicated index (for example, set `MEILISEARCH_INDEX` to
`kenhub_semantic_bge_m3_int8_v3`) and fully resubmitted from Rails. Do not change
an old index's embedder settings to bypass the compatibility check.
Automatic document splitting does not change embeddings of previously valid
chunks, so it needs no new preprocessing identity or reindex of those records.
Previously rejected long documents must be resubmitted to index them.

All embeddings are computed before submitting a whole-document replacement.
The service waits for the remote task to succeed before returning HTTP 200 or
204. A failed task returns an error and preserves the previous document.
A timeout or uncertain write disables service readiness and further store
operations; restart after resolving the remote task. Startup waits for
enqueued/processing tasks on the dedicated index before checking compatibility.
Meilisearch may still complete a write after the caller times out; Rails must
retry idempotently. Do not write to this index from other services.
Queued operations recheck readiness under the storage lock. Client
cancellation does not release that lock before the remote operation finishes.
All document modifications share a FIFO mutation lock covering inference
and remote commit: a newer DELETE cannot overtake an older PUT's encoding.
This intentionally serializes modifications across documents as well; searches
can still read the previously committed index while replacement encoding runs.
Inference waits on an async lock before creating a worker-thread task.
Cancelled waiters never start inference. Started inference finishes before
releasing its lock and request capacity, even on repeated cancellation;
a replacement cancelled during inference is not submitted to Meilisearch.
The admission limit uses pure ASGI middleware, not a detached downstream
request task: an HTTP cancellation retains its slot until active inference or
store work finishes. Overload still returns 429 with `Retry-After: 1`;
health endpoints bypass admission. Body/validation errors and backend failures
release their slots once processing finishes.
Body uploads have a 10-second idle timeout (`REQUEST_BODY_IDLE_TIMEOUT`) and
a 30-second total deadline (`REQUEST_BODY_TIMEOUT`), both configurable finite
positive seconds. Incomplete uploads return **408** with error code
`request_timeout`, close the connection and release admission capacity.
The total deadline prevents a slow trickle from extending the upload forever.
These limits apply only to body reception, not model inference or indexing;
health checks bypass them.
Non-positive `MAX_CONCURRENT_REQUESTS`, `MAX_SEARCH_CANDIDATES` and
`MAX_REQUEST_BYTES` fail configuration loading rather than starting an
apparently healthy service that rejects all requests or returns empty results.
Transient read/probe connection failures and remote 502/503/504 return 503 and
can recover without a restart. A transient health failure is not latched;
subsequent probes retry. Incompatible index settings and ambiguous writes
remain fail-closed. Further business requests in that state return 503 rather
than 500. A definite failed indexing task remains a 500 operation failure.
An explicit HTTP 4xx rejection (except 408) clears pending-write state and
returns an operation error, without permanently disabling healthy searches.
Transport failures, 408, server failures and invalid task responses remain
ambiguous. Invalid JSON/UTF-8 body parsing returns the same 422
`invalid_request` envelope as field validation.

Meilisearch owns persistence and atomic document updates. Back up its data
and settings with its supported snapshot/dump procedures. Keep the pinned
model available for reproducible indexing. Run one Python worker/container
per dedicated index to preserve local coordination.

Existing LanceDB/SQLite files are not read, migrated, or deleted. Populate the
new Meilisearch index by resubmitting source documents from Rails; do not import
OpenAI/Pinecone vectors. Changing models requires a separate new index and
full resubmission.

## Rails integration contract

Rails and Sidekiq call this service directly over the private network. The
service has no internal JWT requirement. Use finite connect/read timeouts in
Rails and keep its upstream timeout shorter than the Rails/MCP caller timeout.
Indexing timeouts must account for embedding plus remote task completion
(`MEILISEARCH_TASK_TIMEOUT` defaults to 120 seconds); use Sidekiq for writes.
Rails should resolve `type` and `record_id` against currently published
records, enforce locale and access rules, and preserve the search result order.
The existing Rails locale restriction remains in force: Python accepts `zh`
and `en`, but Rails access to English still depends on Rails' existing policy.

`ai:reindex` should be treated as a full resubmission of currently published
records. The document API replaces each submitted record's chunks, but a
reindex does not by itself discover and clear stale IDs absent from Rails.

## Operations and known limits

### Measured image and memory footprint

The following **pre-review** measurements were taken on 2026-11-06 with the
built int8 ONNX image, Docker Desktop on Linux,
one Uvicorn worker, a 4-CPU quota and 4 GiB container limit. The daemon exposes
20 logical CPUs; this is not a benchmark on a dedicated four-core machine.
Tests used a temporary Meilisearch 1.16.0 index and synthetic English anatomy
text, not a production corpus.

- Pre-review local uncompressed image: **981,812,938 bytes**
  (982 MB / 936 MiB).
- Bundled model graph: **568,479,395 bytes**; complete model directory about
  559 MiB. The production virtualenv occupies about 221 MiB.
- Memory was sampled every 50 ms using a lightweight stdlib-only client.
  Python RSS excludes the sampler; container figures include it and charged
  file cache. The following workloads ran sequentially in one fresh service,
  so later measurements include retained allocations from earlier requests.

| Workload | Python RSS peak | Container working-set peak |
| --- | ---: | ---: |
| Ready, idle | 941 MiB | 1.01 GiB |
| Five short queries | 944 MiB | 1.01 GiB |
| Index one 273-token chunk | 962 MiB | 1.01 GiB |
| Index eight 273-token chunks | 1082 MiB | 1.13 GiB |
| Index one 1029-token chunk | 1191 MiB | 1.23 GiB |
| Index one 2055-token chunk | 1760 MiB | 1.79 GiB |
| Two concurrent short queries after indexing | 1760 MiB | 1.79 GiB |

The container's lifetime cgroup peak was about 1.79 GiB; no OOM events occurred.
After indexing, idle RSS remained about 1760 MiB: returning a response does
not imply the inference runtime returns all allocated memory to the OS.
The existing Meilisearch container was around 251-263 MiB during these small
tests; this is its total observed usage, not incremental vector-index cost or
a forecast for the production corpus.

The reviewed implementation no longer pads/runs all chunks together. It
validates all input lengths, then performs **single-text inference batches**
in input order; pooling is serialized too. Intermediate activation memory
therefore does not scale with the number of chunks in a document. A document
replacement is still submitted only after every embedding succeeds.

A post-fix isolated test of one legal **8192-token** chunk exceeded a hard
**4 GiB** container limit (exit 137). Single-text batching alone is therefore
not sufficient for that context length. The default `MAX_MODEL_TOKENS=2048`
keeps every inference within that budget. Python now automatically splits
longer document chunks; query overflow is still rejected with 422, and no
text is silently truncated. Compose limits the service to four CPUs and
4 GiB without swap.
An isolated 2048-token inference completed under that limit, with about
1500 MiB peak process RSS; charged container memory varies with file cache.
In a second isolated offline run, one 2048-token chunk followed by a full
256-chunk document (273 tokens per chunk) peaked at 1503 MiB process RSS and
1485 MiB cgroup memory, with zero OOM events. In that four-CPU-quota container,
the long chunk took 14.8s and the 256-chunk request took 270.6s; these are
synthetic encoder-only timings, not API latency or representative throughput.
The automatic-splitting image is **981,829,980 bytes** (982 MB / 936 MiB) locally.
In an isolated offline four-CPU/4 GiB container, a 6000-character Chinese input
split into four chunks and encoded successfully, with 1249 MiB peak process
RSS, 1291 MiB peak cgroup memory, and no OOM events. This is one synthetic
document, not a guarantee for every allowed request or API latency.
The earlier CLS-pooling image was **981,825,552 bytes** locally,
including the model. Its real FastAPI/Meilisearch smoke test passed under the
same 4 GiB/four-CPU limits, including startup, writes/search/delete, token
overflow 422 and streamed-body overflow 413.
Do not treat old all-batch measurements as current throughput measurements.
The worst-case 256-by-2048 inference shape is covered by regression tests,
but its complete end-to-end latency has not been measured. Sequential long
chunks can take considerable CPU time; use Sidekiq and realistic timeouts.

`MAX_MODEL_TOKENS` can be explicitly raised up to the model's 8192 ceiling,
but also increase the container memory allocation and validate the exact
longest input on target hardware before doing so. 4 GiB is not a guarantee for
larger limits or arbitrary future models.
Raw float32 vector payload alone is 4096 bytes per chunk; Meilisearch storage,
text, index structures and memory mapping add further overhead.

- Search uses Meilisearch's approximate vector retrieval with
  `semanticRatio: 1`; keyword/hybrid ranking is not enabled. Candidate recall
  and latency can differ from the former exact LanceDB scan.
- Returned `score` is recomputed from the retrieved vectors as the maximum
  chunk cosine similarity in [-1, 1], not Meilisearch's ranking score.
- Inference is synchronous but runs outside the async event loop. Requests are
  bounded; overload returns `429` with `Retry-After: 1`.
- One document holds all its chunk vectors, so chunks cannot consume multiple
  result slots. `MAX_SEARCH_CANDIDATES` now caps document candidates, not chunk
  rows; no adaptive chunk expansion is needed.
- Scoring retrieves candidate chunk vectors. Large chunk counts increase
  response traffic between Meilisearch and Python; measure it on real content.
- The default compose setup is CPU-only. No production model-load time,
  memory, indexing throughput, retrieval quality, or p50/p95 latency is
  claimed until measured on representative hardware and content.
- The production encoder uses BGE-M3's dense output, CLS pooling,
  1024 dimensions, and up to 8192 tokens. Sparse and ColBERT outputs are not
  used.
- A local comparison of the full-precision and int8 ONNX encoders over
  `tests/data/anatomy_retrieval.jsonl` (11 curated English/Chinese queries)
  was rerun after correcting CLS pooling, with queries encoded separately
  from documents and single-text inference batches. Both produced Recall@1
  0.909, Recall@3 1.000, MRR 0.955, and mean positive rank 1.091.
  Full precision took 15.1s and int8 5.2s on the
  development host, including model loading. Actual Meilisearch 1.16.0
  retrieval reproduced those metrics using temporary isolated candidates.
  Multi-chunk benchmark documents are scored by their best chunk, just like
  production. This is a smoke test,
  not evidence of equivalent quality on the Kenhub corpus; validate with
  representative production queries and documents before relying on the
  ranking quality.
- Model alternatives considered:

  | Option | Weights | Dimensions / context | Trade-off |
  | --- | ---: | --- | --- |
  | BGE-M3 PyTorch fp32 | 2.27 GB | 1024 / 8192 | Original reference; large runtime and weights. |
  | BGE-M3 ONNX fp32 | about 2.27 GB | 1024 / 8192 | Can remove PyTorch, but does not materially shrink the model weights. |
  | BGE-M3 ONNX int8 (selected) | 568 MB | 1024 / 8192 | Same base encoder and shape; small-set ranking matched fp32, but corpus-level quality remains unverified. |
  | multilingual-e5-small int8 ONNX | about 118 MB for an AVX512-VNNI build | 384 / 512 | Much smaller, but different model, shorter context, and requires `query:` / `passage:` prefixes; not an equivalent drop-in. |

  The Meilisearch-backed image was built at 982 MB locally, including the
  568 MB model. PyTorch, Sentence Transformers, LanceDB and PyArrow are absent
  from its runtime. This is the local image size, not compressed registry size.

  FastEmbed uses ONNX Runtime but its standard prebuilt model catalog does not
  provide BGE-M3, so this service uses ONNX Runtime directly. FastEmbed could
  be reconsidered if adopting one of its supported smaller models. The E5
  size/context figures and BGE-M3 model specifications come from their
  published model cards and converted model files; measure actual image size
  and quality in the deployment environment.
- To repeat the local comparison, install the optional evaluation dependencies
  and download the pinned full-precision BGE-M3 files into
  `cache/models/bge-m3`:

  ```bash
  uv sync --group eval
  uvx hf download BAAI/bge-m3 \
    config.json modules.json sentence_bert_config.json \
    config_sentence_transformers.json 1_Pooling/config.json \
    pytorch_model.bin tokenizer.json tokenizer_config.json \
    special_tokens_map.json sentencepiece.bpe.model \
    --revision 5617a9f61b028005a4858fdac845db406aefb181 \
    --local-dir cache/models/bge-m3
  uv run --group eval python -m scripts.benchmark_embedding
  # Also verify retrieval against Meilisearch, with configured credentials:
  uv run --group eval --env-file .env \
    python -m scripts.benchmark_embedding --meilisearch
  ```
  The remote benchmark creates/deletes only a random
  `semantic_search_benchmark_*` index. It requests all candidates in each small
  case, so it does not measure large-corpus ANN top-k recall.
- An opt-in real-model test is available with
  `RUN_REAL_MODEL_TESTS=1 MODEL_PATH=/path/to/bge-m3-onnx-int8 uv run pytest
  tests/test_real_model.py`; it requires the model already present locally and
  keeps Hugging Face in offline mode.
- Run the real ONNX + Meilisearch integration test with a configured
  environment:

  ```bash
  RUN_REAL_MEILISEARCH_TESTS=1 uv run --env-file .env \
    pytest -q tests/test_real_meilisearch.py
  ```

  It creates a randomly named `semantic_search_test_*` index, verifies
  write/search/replace/restart/delete, then removes only that index. The API
  key needs permissions for these temporary indexes as well.
- Dynamic int8 quantization can produce different vectors in different batch
  contexts. Single-text batching removes neighboring-text dependence; an
  opt-in real-model test verifies identical vectors when encoding texts
  together through the service or in separate calls.
- Errors and default logs do not include request bodies, chunk text, queries,
  model credentials, or internal exception details in HTTP responses.
