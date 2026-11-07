"""Local, offline BGE-M3 int8 ONNX embedding implementation."""

import logging
import os
import re
import threading
from collections.abc import Sequence

import numpy as np
from tokenizers import Tokenizer

from api.settings import Settings

LOGGER = logging.getLogger("semantic_search.embedding")
EXPECTED_DIMENSION = 1024
EXPECTED_MAX_TOKENS = 8192
ONNX_MODEL_FILE = "onnx/model_int8.onnx"
MAX_DOCUMENT_CHUNKS = 256


class EmbeddingError(ValueError):
    """Raised when text cannot be embedded without truncation."""


class LocalBgeEmbedding:
    """Load a local quantized BGE-M3 ONNX model and embed text."""

    def __init__(self, settings: Settings):
        """Load and validate the model and tokenizer from disk."""
        if settings.embedding_dimension != EXPECTED_DIMENSION:
            raise ValueError(
                f"BGE-M3 requires dimension {EXPECTED_DIMENSION}, "
                f"got {settings.embedding_dimension}"
            )
        if settings.device != "cpu":
            raise ValueError("The ONNX embedding backend supports CPU only")
        if not settings.model_path.is_dir():
            raise FileNotFoundError(
                f"Local embedding model directory not found: "
                f"{settings.model_path}"
            )

        model_path = settings.model_path / ONNX_MODEL_FILE
        tokenizer_path = settings.model_path / "tokenizer.json"
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Local ONNX model file not found: {model_path}"
            )
        if not tokenizer_path.is_file():
            raise FileNotFoundError(
                f"Local tokenizer file not found: {tokenizer_path}"
            )

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Delay this heavyweight import until runtime startup.
        # pylint: disable=import-outside-toplevel
        import onnxruntime as ort

        LOGGER.info("Loading local ONNX embedding model from %s", model_path)
        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
            sess_options=options,
        )
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()
        self._pad_token_id = self._tokenizer.token_to_id("<pad>")
        if self._pad_token_id is None:
            raise ValueError("The ONNX tokenizer has no <pad> token")
        self._input_names = {
            item.name for item in self._session.get_inputs()
        }
        required_inputs = {"input_ids", "attention_mask"}
        if not required_inputs.issubset(self._input_names):
            raise ValueError(
                f"ONNX model inputs {sorted(self._input_names)} do not "
                f"include {sorted(required_inputs)}"
            )
        self._max_tokens = min(
            settings.max_model_tokens, EXPECTED_MAX_TOKENS
        )
        self._dimension = settings.embedding_dimension
        self._lock = threading.Lock()
        if self._max_tokens < 1:
            raise ValueError("The configured model token limit must be > 0")
        probe = self.embed(["a"])
        if not np.allclose(np.linalg.norm(probe, axis=1), 1.0, atol=1e-5):
            raise RuntimeError("Model startup probe is not normalized")

    def validate_text(self, text: str) -> None:
        """Reject text longer than the model limit instead of truncating."""
        token_count = len(
            self._tokenizer.encode(text, add_special_tokens=True).ids
        )
        if token_count > self._max_tokens:
            raise EmbeddingError(
                f"Text contains {token_count} tokens; limit is "
                f"{self._max_tokens}"
            )

    def split_document(self, texts: Sequence[str]) -> list[str]:
        """Split oversized inputs without decoding tokens or losing text."""
        chunks = []
        pending = list(reversed(texts))
        while pending:
            text = pending.pop()
            try:
                self.validate_text(text)
            except EmbeddingError:
                first = len(text) - len(text.lstrip())
                last = len(text.rstrip())
                middle = (first + last) // 2
                boundaries = [
                    match.end() for match in re.finditer(
                        r"\n+|[.!?。！？；;]\s*", text
                    )
                    if first < match.end() < last
                    and len(text) // 4 <= match.end() <= 3 * len(text) // 4
                ]
                split = min(
                    boundaries, key=lambda pos: abs(pos - middle)
                ) if boundaries else middle
                if not first < split < last:
                    raise EmbeddingError(
                        "Text cannot be split within the model token limit"
                    ) from None
                pending.extend([text[split:], text[:split]])
            else:
                chunks.append(text)
            if len(chunks) + len(pending) > MAX_DOCUMENT_CHUNKS:
                raise EmbeddingError(
                    "Document exceeds 256 chunks after automatic splitting"
                )
        return chunks

    def _make_inputs(
        self, texts: Sequence[str]
    ) -> dict[str, np.ndarray]:
        """Tokenize and pad a batch into the ONNX graph inputs."""
        encodings = self._tokenizer.encode_batch(list(texts))
        max_length = max(len(item.ids) for item in encodings)
        input_ids = []
        attention_masks = []
        token_type_ids = []
        for item in encodings:
            padding = max_length - len(item.ids)
            input_ids.append(item.ids + [self._pad_token_id] * padding)
            attention_masks.append(item.attention_mask + [0] * padding)
            token_type_ids.append(item.type_ids + [0] * padding)

        feeds = {
            "input_ids": np.asarray(input_ids, dtype=np.int64),
            "attention_mask": np.asarray(
                attention_masks, dtype=np.int64
            ),
        }
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.asarray(
                token_type_ids, dtype=np.int64
            )
        return feeds

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return finite, normalized float32 vectors for input texts."""
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)
        for text in texts:
            self.validate_text(text)

        # Single-text batches bound activation memory and remove dependence
        # on neighboring chunks in dynamically quantized inference.
        with self._lock:
            return np.concatenate(
                [self._embed_one(text) for text in texts], axis=0
            )

    def _embed_one(self, text: str) -> np.ndarray:
        """Encode one validated text while holding the inference lock."""
        feeds = self._make_inputs([text])
        outputs = self._session.run(None, feeds)
        hidden = np.asarray(outputs[0], dtype=np.float32)
        if hidden.ndim != 3 or hidden.shape[:2] != feeds["input_ids"].shape:
            raise RuntimeError(
                f"ONNX model returned unexpected shape {hidden.shape}"
            )
        if hidden.shape[2] != self._dimension:
            raise RuntimeError(
                f"ONNX model returned dimension {hidden.shape[2]}; "
                f"expected {self._dimension}"
            )

        embeddings = hidden[:, 0, :]
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if (
            np.any(norms == 0) or not np.isfinite(norms).all()
            or not np.isfinite(embeddings).all()
        ):
            raise RuntimeError("Embedding returned an invalid vector")
        result = embeddings / norms
        if result.shape != (1, self._dimension):
            raise RuntimeError(
                f"Embedding returned shape {result.shape}; expected "
                f"(1, {self._dimension})"
            )
        return np.asarray(result, dtype=np.float32)
