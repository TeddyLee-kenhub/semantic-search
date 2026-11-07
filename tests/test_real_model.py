"""Opt-in integration test for an installed local BGE-M3 ONNX model."""

import os
from pathlib import Path

import numpy as np
import pytest

from api.embedding import EmbeddingError, LocalBgeEmbedding
from api.settings import Settings


@pytest.mark.skipif(
    os.getenv("RUN_REAL_MODEL_TESTS") != "1",
    reason="set RUN_REAL_MODEL_TESTS=1 to load the real local model",
)
def test_real_model_loads_offline_and_returns_normalized_1024d_vectors():
    """Load local weights only and verify the actual embedding output."""
    settings = Settings.from_env()
    assert settings.model_path.is_dir()
    embedder = LocalBgeEmbedding(settings)
    vectors = embedder.embed(["cranial anatomy", "颅骨解剖"])

    assert vectors.shape == (2, 1024)
    assert np.isfinite(vectors).all()
    assert np.linalg.norm(vectors, axis=1) == pytest.approx([1.0, 1.0])
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert Path(settings.model_path).is_dir()
    separate = np.concatenate([
        embedder.embed(["cranial anatomy"]),
        embedder.embed(["颅骨解剖"]),
    ])
    np.testing.assert_array_equal(vectors, separate)
    # pylint: disable=protected-access
    hidden = embedder._session.run(
        None, embedder._make_inputs(["cranial anatomy"])
    )[0]
    cls = hidden[:, 0, :]
    cls /= np.linalg.norm(cls, axis=1, keepdims=True)
    np.testing.assert_allclose(vectors[:1], cls, atol=1e-6)
    for text in [
        "股" * 6000,
        ("The femur is the thigh bone. " * 500),
        ("股骨是大腿骨。\nThe femur is the thigh bone. 😀 " * 200),
    ]:
        chunks = embedder.split_document([text])
        assert "".join(chunks) == text
        assert len(chunks) > 1
        assert all(chunk.strip() for chunk in chunks)
        for chunk in chunks:
            embedder.validate_text(chunk)
    with pytest.raises(EmbeddingError):
        embedder.embed(["股" * 2046])
    with pytest.raises(EmbeddingError, match="256"):
        embedder.split_document(["股" * 6000] * 256)
