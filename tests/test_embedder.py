import numpy as np
import pytest
from rag_pipeline.embedder import Embedder

@pytest.fixture(scope="module")
def embedder():
    """
    scope="module" means this fixture is created once for the entire
    test file, not once per test. Since loading the model is slow,
    we share one instance across all tests in this file.
    """
    return Embedder()


def test_embed_correct_shape(embedder):
    vector = embedder.embed("this is a test sentence")
    assert vector.shape == (384,), f"expected shape (384,) got {vector.shape}"

def test_embed_returns_numpy_array(embedder):
    vector = embedder.embed("Hello World")
    assert isinstance(vector, np.ndarray)

def test_embed_batch_returns_correct_shape(embedder):
    vector = embedder.embed_batch(["one sentence", "two sentence", "three sentence"])
    assert vector.shape == (3,384), f"expected shape (3,384) got {vector.shape}"

def test_embed_match_single_item(embedder):
    vector = embedder.embed_batch(["single sentence"])
    assert vector.shape == (1,384), f"expected shape (1,384) got {vector.shape}"

def test_similar_texts_have_high_similarity(embedder):
    vector_1 = embedder.embed("the cat sat on the mat")
    vector_2 = embedder.embed("a cat is sitting on the mat")

    # Cosine similarity = dot product of normalised vectors
    v1_norm = vector_1 / np.linalg.norm(vector_1)
    v2_norm = vector_2 / np.linalg.norm(vector_2)
    cosine_sim = np.dot(v1_norm, v2_norm)

    assert cosine_sim > 0.8, f"expected high similarity, got {cosine_sim : .3f}"

def test_different_texts_have_low_similarity(embedder):
    vector_1 = embedder.embed("the cat sat on the mat")
    vector_2 = embedder.embed("quantum mechanics describes subatomic particles")

    # Cosine similarity = dot product of normalised vectors
    v1_norm = vector_1 / np.linalg.norm(vector_1)
    v2_norm = vector_2 / np.linalg.norm(vector_2)
    cosine_sim = np.dot(v1_norm, v2_norm)

    assert cosine_sim < 0.5, f"expected low similarity, got {cosine_sim : .3f}"


