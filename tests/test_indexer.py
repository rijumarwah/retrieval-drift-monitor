import uuid
import numpy as np
import pytest
from unittest.mock import MagicMock
from rag_pipeline.indexer import Indexer, CLASS_NAME


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.dimension = 384
    embedder.embed_batch.side_effect = lambda texts: np.random.rand(len(texts), 384)
    return embedder

@pytest.fixture
def mock_client():
    return MagicMock()

@pytest.fixture
def indexer(mock_embedder, mock_client):
    return Indexer(
        embedder=mock_embedder,
        client=mock_client,
        chunk_size=100,
        chunk_overlap=20,
    )

def test_chunk_text_produces_correct_overlap(indexer):
    text = "a" * 200
    chunks = indexer._chunk_text(text)
    assert len(chunks) == 3

def test_chunk_text_no_empty_chunks(indexer):
    text = "Hello world. " * 10
    chunks = indexer._chunk_text(text)
    for chunk in chunks:
        assert chunk.strip() != ""

def test_chunk_text_single_chunk_when_text_is_short(indexer):
    text = "Short text."
    chunks = indexer._chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0] == "Short text."

def test_index_document_returns_chunk_ids(indexer):
    chunk_ids = indexer.index_document(
        text="word " * 200,
        source="test_doc"
    )
    assert isinstance(chunk_ids, list)
    assert len(chunk_ids) > 0
    for cid in chunk_ids:
        uuid.UUID(cid)

def test_setup_schema_skips_if_exists(indexer, mock_client):
    mock_client.schema.exists.return_value = True
    indexer.setup_schema()
    mock_client.schema.create_class.assert_not_called()

def test_setup_schema_creates_if_not_exists(indexer, mock_client):
    mock_client.schema.exists.return_value = False
    indexer.setup_schema()
    mock_client.schema.create_class.assert_called_once()
