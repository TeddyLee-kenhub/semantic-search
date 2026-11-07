FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_NO_DEV=1 \
    UV_COMPILE_BYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    MODEL_PATH=/models/bge-m3-int8 \
    EMBEDDING_DEVICE=cpu

RUN useradd --create-home --uid 1000 app
WORKDIR /app
RUN chown app:app /app
USER app

COPY --chown=app:app pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

COPY --chown=app:app main.py ./
COPY --chown=app:app api/ api/
COPY --chown=app:app cache/models/bge-m3-onnx-int8/ /models/bge-m3-int8/

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "python", "main.py"]
