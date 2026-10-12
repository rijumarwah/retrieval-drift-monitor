import json
import pytest
from pathlib import Path
from eval.golden_set_loader import load_golden_set, GoldenQuery


VALID_ENTRY = {
    "query_id": "q001",
    "question": "What is the capital of France?",
    "expected_answer": "Paris",
    "correct_chunk_id": "abc-123",
    "source": "wiki_france",
}

@pytest.fixture
def golden_set_file(tmp_path):
    """
    tmp_path is a built-in pytest fixture that gives you a
    temporary directory that is cleaned up after the test.
    """
    path = tmp_path / "golden_set.json"
    path.write_text(json.dumps([VALID_ENTRY]))
    return path

def test_load_returns_list_of_golden_queries(golden_set_file):
    queries = load_golden_set(golden_set_file)
    assert isinstance(queries, list)
    assert len(queries) == 1
    assert isinstance(queries[0], GoldenQuery)

def test_load_maps_fields_correctly(golden_set_file):
    query = load_golden_set(golden_set_file)[0]
    assert query.query_id == "q001"
    assert query.question == "What is the capital of France?"
    assert query.correct_chunk_id == "abc-123"
    assert query.source == "wiki_france"

def test_load_raises_if_file_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_golden_set(tmp_path / "nonexistent.json")

def test_load_raises_if_not_list(tmp_path):
    path = tmp_path / "golden_set.json"
    path.write_text(json.dumps({"not": "a list"}))
    with pytest.raises(ValueError, match="must be a JSON array"):
        load_golden_set(path)

def test_load_raises_if_empty(tmp_path):
    path = tmp_path / "golden_set.json"
    path.write_text(json.dumps([]))
    with pytest.raises(ValueError, match="empty"):
        load_golden_set(path)

def test_load_raises_if_missing_fields(tmp_path):
    path = tmp_path / "golden_set.json"
    incomplete = {"query_id": "q001", "question": "What?"}
    path.write_text(json.dumps([incomplete]))
    with pytest.raises(ValueError, match="missing required fields"):
        load_golden_set(path)
