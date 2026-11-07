"""Tests for local ONNX embedding input validation and output."""

import threading
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
import pytest

from api.embedding import EmbeddingError, LocalBgeEmbedding
from api.settings import Settings


class FakeTokenizer:
    """Simple tokenizer substitute for input validation and batching."""

    # This fake matches the Tokenizers methods exercised by the tests.
    # pylint: disable=too-few-public-methods
    def no_truncation(self):
        """Match disabling tokenizer-level truncation at startup."""

    def no_padding(self):
        """Match disabling tokenizer-level padding at startup."""

    def token_to_id(self, token):
        """Return the fake pad token ID."""
        assert token == "<pad>"
        return 0

    def encode(self, text, *, add_special_tokens):
        """Return whitespace-delimited tokens."""
        assert add_special_tokens
        return FakeEncoding(list(range(len(text.split()) + 2)))

    def encode_batch(self, texts):
        """Return tokenized records with attention and type masks."""
        return [self.encode(text, add_special_tokens=True) for text in texts]


class FakeEncoding:
    """Minimal token encoding with padding-related fields."""

    # pylint: disable=too-few-public-methods
    def __init__(self, ids):
        self.ids = ids
        self.attention_mask = [1] * len(ids)
        self.type_ids = [0] * len(ids)


class FakeSession:
    """ONNX session substitute returning fixed hidden states."""

    # pylint: disable=too-few-public-methods
    def run(self, _outputs, feeds):
        """Return hidden states matching the provided token batch."""
        batch, sequence = feeds["input_ids"].shape
        return [np.ones((batch, sequence, 1024), dtype=np.float32)]


class RecordingSession(FakeSession):
    """Record inference batch shapes without loading a model."""

    # pylint: disable=too-few-public-methods
    def __init__(self):
        self.shapes = []

    def run(self, outputs, feeds):
        """Record each graph invocation and return fake hidden states."""
        self.shapes.append(feeds["input_ids"].shape)
        return super().run(outputs, feeds)


def fake_embedding(session=None):
    """Create a local encoder with test dependencies and production limits."""
    # pylint: disable=protected-access
    embedder = object.__new__(LocalBgeEmbedding)
    embedder._tokenizer = FakeTokenizer()
    embedder._pad_token_id = 0
    embedder._input_names = {"input_ids", "attention_mask"}
    embedder._session = session or FakeSession()
    embedder._max_tokens = 8192
    embedder._dimension = 1024
    embedder._lock = threading.Lock()
    return embedder


def test_maximum_document_uses_single_text_inference_batches():
    """Do not allocate a hidden-state tensor for all 256 long chunks."""
    session = RecordingSession()
    embedder = fake_embedding(session)
    text = "word " * 2046
    vectors = embedder.embed([text] * 256)
    assert vectors.shape == (256, 1024)
    assert session.shapes == [(1, 2048)] * 256


def test_long_and_short_chunks_are_not_padded_together():
    """Preserve input order without padding short chunks to the longest."""
    session = RecordingSession()
    embedder = fake_embedding(session)
    embedder.embed(["one", "word " * 8190, "one two"])
    assert session.shapes == [(1, 3), (1, 8192), (1, 4)]


def test_all_chunks_are_validated_before_inference():
    """An oversized later chunk cannot trigger partial inference first."""
    session = RecordingSession()
    embedder = fake_embedding(session)
    with pytest.raises(EmbeddingError):
        embedder.embed(["valid", "word " * 8191])
    assert not session.shapes


@pytest.mark.parametrize("fault", ["dimension", "zero", "nan", "runtime"])
def test_startup_probe_rejects_invalid_model(
    settings, tmp_path, monkeypatch, fault
):
    """Readiness cannot pass until real graph inference has been validated."""
    class InvalidSession:
        """Minimal session returning a selected invalid inference result."""

        def get_inputs(self):
            """Report the required graph inputs."""
            return [
                SimpleNamespace(name="input_ids"),
                SimpleNamespace(name="attention_mask"),
            ]

        def run(self, _outputs, feeds):
            """Produce the injected graph failure."""
            if fault == "runtime":
                raise RuntimeError("inference failed")
            dimension = 384 if fault == "dimension" else 1024
            value = 0 if fault == "zero" else float("nan")
            shape = (*feeds["input_ids"].shape, dimension)
            return [np.full(shape, value, dtype=np.float32)]

    (tmp_path / "onnx").mkdir()
    (tmp_path / "onnx/model_int8.onnx").touch()
    (tmp_path / "tokenizer.json").touch()
    monkeypatch.setattr(
        ort, "InferenceSession", lambda *a, **kw: InvalidSession()
    )
    monkeypatch.setattr(
        "api.embedding.Tokenizer.from_file", lambda _: FakeTokenizer()
    )
    with pytest.raises(RuntimeError):
        LocalBgeEmbedding(replace(settings, model_path=tmp_path))


def test_validate_text_rejects_text_that_would_be_truncated():
    """Reject model inputs above the configured token maximum."""
    # pylint: disable=protected-access
    embedder = object.__new__(LocalBgeEmbedding)
    embedder._tokenizer = FakeTokenizer()
    embedder._max_tokens = 4
    with pytest.raises(EmbeddingError, match="limit is 4"):
        embedder.validate_text("one two three")


def test_default_limit_bounds_inference_without_changing_model_ceiling(
    monkeypatch
):
    """The default deployment rejects long inputs, with explicit opt-in."""
    monkeypatch.delenv("MAX_MODEL_TOKENS", raising=False)
    assert Settings.from_env().max_model_tokens == 2048
    monkeypatch.setenv("MAX_MODEL_TOKENS", "8192")
    assert Settings.from_env().max_model_tokens == 8192


def test_embed_returns_normalized_float32_vectors():
    """Normalize the ONNX CLS output to the expected shape."""
    # pylint: disable=protected-access
    embedder = object.__new__(LocalBgeEmbedding)
    embedder._tokenizer = FakeTokenizer()
    embedder._pad_token_id = 0
    embedder._input_names = {"input_ids", "attention_mask"}
    embedder._session = FakeSession()
    embedder._max_tokens = 8
    embedder._dimension = 1024
    embedder._lock = threading.Lock()

    vectors = embedder.embed(["one", "one two"])

    assert vectors.shape == (2, 1024)
    assert vectors.dtype == np.float32
    assert np.linalg.norm(vectors, axis=1) == pytest.approx([1.0, 1.0])


def test_embedding_uses_cls_not_mean_pooling():
    """Non-CLS tokens must not contribute to the dense BGE-M3 embedding."""
    class DifferentTokens(FakeSession):
        """Return distinct CLS and other token directions."""

        # pylint: disable=too-few-public-methods
        def run(self, _outputs, feeds):
            """Make mean pooling observably different from CLS pooling."""
            hidden = np.zeros(
                (*feeds["input_ids"].shape, 1024), dtype=np.float32
            )
            hidden[:, 0, 0] = 3
            hidden[:, 0, 1] = 4
            hidden[:, 1:, 2] = 10
            return [hidden]

    vectors = fake_embedding(DifferentTokens()).embed(["one two"])
    assert vectors[0, :3] == pytest.approx([0.6, 0.8, 0.0])


@pytest.mark.parametrize("text", [
    "one two three four five six seven eight",
    "one two.\nthree four! five six? seven eight",
    "  one two three four five six seven eight  ",
])
def test_split_document_preserves_text_and_token_budget(text):
    """Use exact source substrings and keep every piece within the limit."""
    encoder = fake_embedding()
    encoder._max_tokens = 5  # pylint: disable=protected-access
    chunks = encoder.split_document([text])
    assert len(chunks) > 1
    assert "".join(chunks) == text
    assert all(chunk.strip() for chunk in chunks)
    for chunk in chunks:
        encoder.validate_text(chunk)


def test_split_document_preserves_valid_chunks_and_empty_document():
    """Do not change embeddings or boundaries for already-valid input."""
    encoder = fake_embedding()
    assert encoder.split_document(["one", "two"]) == ["one", "two"]
    assert not encoder.split_document([])


def test_split_document_rejects_expanded_count_before_inference():
    """Apply the same 256 limit to expanded chunks, not only input chunks."""
    session = RecordingSession()
    encoder = fake_embedding(session)
    encoder._max_tokens = 4  # pylint: disable=protected-access
    assert len(encoder.split_document(["one two"] * 256)) == 256
    with pytest.raises(EmbeddingError, match="256"):
        encoder.split_document(["one two three four"] * 129)
    assert not session.shapes


def test_split_document_rejects_unsplittable_token_budget():
    """A budget too small for a single character must fail without looping."""
    encoder = fake_embedding()
    encoder._max_tokens = 2  # pylint: disable=protected-access
    with pytest.raises(EmbeddingError, match="cannot be split"):
        encoder.split_document([" x "])
