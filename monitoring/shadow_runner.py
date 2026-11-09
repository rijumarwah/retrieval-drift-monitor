import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

from eval.golden_set_loader import GoldenQuery


@dataclass
class QueryResult:
    """
    Raw outcome of one golden query in one run. No metrics here,
    only what happened. The metric engine interprets it.
    """
    run_id: str
    query_id: str
    timestamp: datetime
    correct_chunk_id: str
    retrieved_chunk_ids: list[str]  # in rank order, best first
    scores: list[float]
    error: str | None = None        # set if the request failed


class ShadowRunner:
    def __init__(
        self,
        base_url: str,
        golden_queries: list[GoldenQuery],
        top_k: int = 5,
        http_client: httpx.Client | None = None,
    ):
        self.golden_queries = golden_queries
        self.top_k = top_k
        # Injectable client: tests pass one with a fake transport,
        # production uses a real one with a timeout.
        self.client = http_client or httpx.Client(
            base_url=base_url, timeout=10.0
        )

    def run_once(self) -> list[QueryResult]:
        """
        Runs every golden query once. All results share one run_id
        and timestamp so they can be grouped as a single evaluation.
        """
        run_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)
        return [
            self._run_query(q, run_id, timestamp)
            for q in self.golden_queries
        ]

    def _run_query(
        self, query: GoldenQuery, run_id: str, timestamp: datetime
    ) -> QueryResult:
        try:
            response = self.client.post(
                "/query",
                json={"question": query.question, "top_k": self.top_k},
            )
            response.raise_for_status()
            results = response.json()["results"]
            return QueryResult(
                run_id=run_id,
                query_id=query.query_id,
                timestamp=timestamp,
                correct_chunk_id=query.correct_chunk_id,
                retrieved_chunk_ids=[r["chunk_id"] for r in results],
                scores=[r["score"] for r in results],
            )
        except (httpx.HTTPError, KeyError, ValueError) as e:
            # One bad query must not abort the whole run.
            return QueryResult(
                run_id=run_id,
                query_id=query.query_id,
                timestamp=timestamp,
                correct_chunk_id=query.correct_chunk_id,
                retrieved_chunk_ids=[],
                scores=[],
                error=str(e),
            )
