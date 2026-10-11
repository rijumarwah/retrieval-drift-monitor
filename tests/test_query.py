import pytest
import numpy as np
from unittest.mock import MagicMock
from rag_pipeline.query import QueryEngine


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.embed.return_value = np.random.rand(384)
    return embedder


@pytest.fixture
def mock_client():
    client = MagicMock()
    # Simulate Weaviate returning 2 chunks
    client.query.get.return_value.with_near_vector.return_value \
        .with_limit.return_value.with_additional.return_value.do.return_value = {
        "data": {
            "Get": {
                "DocumentChunk": [
                    {
                        "chunk_id": "abc-123",
                        "text": "Paris is the capital of France.",
                        "source": "wiki_france",
                        "chunk_index": 0,
                        "_additional": {"certainty": 0.95},
                    },
                    {
                        "chunk_id": "def-456",
                        "text": "France is a country in Western Europe.",
                        "source": "wiki_france",
                        "chunk_index": 1,
                        "_additional": {"certainty": 0.82},
                    },
                ]
            }
        }
    }
    return client


@pytest.fixture
def query_engine(mock_embedder, mock_client):
    return QueryEngine(embedder=mock_embedder, client=mock_client)


def test_query_returns_list(query_engine):
    results = query_engine.query("What is the capital of France?")
    assert isinstance(results, list)


def test_query_returns_correct_number_of_results(query_engine):
    results = query_engine.query("What is the capital of France?", top_k=5)
    assert len(results) == 2  # our mock returns 2


def test_query_result_has_correct_fields(query_engine):
    results = query_engine.query("What is the capital of France?")
    required_fields = {"chunk_id", "text", "source", "chunk_index", "score"}
    for result in results:
        assert required_fields == set(result.keys())


def test_query_result_scores_are_floats(query_engine):
    results = query_engine.query("What is the capital of France?")
    for result in results:
        assert isinstance(result["score"], float)


def test_query_embeds_the_question(query_engine, mock_embedder):
    query_engine.query("What is the capital of France?")
    mock_embedder.embed.assert_called_once_with("What is the capital of France?")
