import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from api.main import app
from rag_pipeline.query import QueryEngine


@pytest.fixture
def client():
    """
    TestClient lets us call FastAPI endpoints without running a server.
    """
    with patch("api.main.Embedder") as mock_embedder_class, \
         patch("api.main.get_weaviate_client") as mock_get_client:

        mock_embedder_class.return_value = MagicMock()
        mock_get_client.return_value = MagicMock()

        with TestClient(app) as c:
            # Override query engine with a controlled mock
            mock_engine = MagicMock(spec=QueryEngine)
            mock_engine.query.return_value = [
                {
                    "chunk_id": "abc-123",
                    "text": "Paris is the capital of France.",
                    "source": "wiki_france",
                    "chunk_index": 0,
                    "score": 0.95,
                }
            ]
            app.state.query_engine = mock_engine
            yield c


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_endpoint_returns_200(client):
    response = client.post("/query", json={"question": "What is the capital of France?"})
    assert response.status_code == 200


def test_query_endpoint_returns_correct_shape(client):
    response = client.post("/query", json={"question": "What is the capital of France?"})
    data = response.json()
    assert "question" in data
    assert "results" in data
    assert isinstance(data["results"], list)


def test_query_endpoint_rejects_empty_question(client):
    response = client.post("/query", json={"question": "   "})
    assert response.status_code == 400


def test_query_endpoint_passes_top_k(client):
    app.state.query_engine.query.return_value = []
    client.post("/query", json={"question": "test", "top_k": 3})
    app.state.query_engine.query.assert_called_once_with(question="test", top_k=3)
