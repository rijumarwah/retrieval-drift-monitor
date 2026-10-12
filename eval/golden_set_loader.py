import json
from pathlib import Path
from dataclasses import dataclass

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"


@dataclass
class GoldenQuery:
    """
    Single entry in the golden query set.

    query_id:       unique identifier, used to track per-query
                    metric trends over time
    question:       the natural language question sent to the pipeline
    expected_answer: the correct answer (used for human review,
                    not for automated scoring)
    correct_chunk_id: the Weaviate chunk_id that contains the answer —
                    this is what the shadow runner checks for in top-k
    source:         which document this came from
    """
    query_id: str
    question: str
    expected_answer: str
    correct_chunk_id: str
    source: str


def load_golden_set(path: Path = GOLDEN_SET_PATH) -> list[GoldenQuery]:
    """
    Loads and validates the golden query set from JSON.
    Raises errors if the file is missing or malformed.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Golden set not found at {path}. "
            "Run scripts/build_corpus.py first, then populate eval/golden_set.json."
        )

    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("golden_set.json must be a JSON array")

    if len(data) == 0:
        raise ValueError("golden_set.json is empty")

    required_fields = {"query_id", "question", "expected_answer",
                       "correct_chunk_id", "source"}

    queries = []
    for i, entry in enumerate(data):
        missing = required_fields - set(entry.keys())
        if missing:
            raise ValueError(
                f"Entry {i} is missing required fields: {missing}"
            )
        queries.append(GoldenQuery(**{k: entry[k] for k in required_fields}))

    return queries
