"""Environment-backed settings for the semantic-search service."""

import os
from dataclasses import dataclass, field
import math
from pathlib import Path

DEFAULT_MODEL_ID = (
    "onnx-community/bge-m3-ONNX"
    "@25b9af8e87a38eb120cfe87125383677b9cd309e:int8"
)


@dataclass(frozen=True)
# Keep configuration immutable and explicit rather than hiding settings in a
# generic dictionary.
# pylint: disable=too-many-instance-attributes
class Settings:
    """Runtime configuration with safe single-instance defaults."""

    model_path: Path
    device: str
    model_id: str
    embedding_dimension: int
    max_model_tokens: int
    max_request_bytes: int
    max_concurrent_requests: int
    max_search_candidates: int
    meilisearch_host: str = "http://127.0.0.1:7700"
    meilisearch_api_key: str = field(default="", repr=False)
    meilisearch_index: str = "kenhub_semantic_bge_m3_int8_v3"
    meilisearch_timeout: float = 10.0
    meilisearch_task_timeout: float = 120.0
    request_body_idle_timeout: float = 10.0
    request_body_timeout: float = 30.0

    def __post_init__(self) -> None:
        """Reject unusable capacities and unbounded remote timeouts."""
        for name in (
            "max_concurrent_requests", "max_search_candidates",
            "max_request_bytes",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")
        for timeout in (
            self.meilisearch_timeout, self.meilisearch_task_timeout,
        ):
            if not math.isfinite(timeout) or timeout <= 0:
                raise ValueError("Meilisearch timeouts must be finite and > 0")
        for name in ("request_body_idle_timeout", "request_body_timeout"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and > 0")

    @classmethod
    def from_env(cls) -> "Settings":
        """Read supported configuration from the process environment."""
        return cls(
            model_path=Path(
                os.getenv("MODEL_PATH", "/models/bge-m3-int8")
            ),
            device=os.getenv("EMBEDDING_DEVICE", "cpu"),
            model_id=os.getenv("EMBEDDING_MODEL_ID", DEFAULT_MODEL_ID),
            embedding_dimension=int(
                os.getenv("EMBEDDING_DIMENSION", "1024")
            ),
            max_model_tokens=int(os.getenv("MAX_MODEL_TOKENS", "2048")),
            max_request_bytes=int(
                os.getenv("MAX_REQUEST_BYTES", str(8 * 1024 * 1024))
            ),
            max_concurrent_requests=int(
                os.getenv("MAX_CONCURRENT_REQUESTS", "2")
            ),
            max_search_candidates=int(
                os.getenv("MAX_SEARCH_CANDIDATES", "100000")
            ),
            meilisearch_host=os.getenv(
                "MEILISEARCH_HOST", "http://127.0.0.1:7700"
            ),
            meilisearch_api_key=os.getenv("MEILISEARCH_API_KEY", ""),
            meilisearch_index=os.getenv(
                "MEILISEARCH_INDEX", "kenhub_semantic_bge_m3_int8_v3"
            ),
            meilisearch_timeout=float(
                os.getenv("MEILISEARCH_TIMEOUT", "10")
            ),
            meilisearch_task_timeout=float(
                os.getenv("MEILISEARCH_TASK_TIMEOUT", "120")
            ),
            request_body_idle_timeout=float(
                os.getenv("REQUEST_BODY_IDLE_TIMEOUT", "10")
            ),
            request_body_timeout=float(
                os.getenv("REQUEST_BODY_TIMEOUT", "30")
            ),
        )
