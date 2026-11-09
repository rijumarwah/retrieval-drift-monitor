import json
import httpx
import pytest
from eval.golden_set_loader import GoldenQuery
from monitoring.shadow_runner import ShadowRunner


def make_queries():
    return [
        GoldenQuery("q001", "Question one?", "A1", "chunk-1", "doc"),
        GoldenQuery("q002", "Question two?", "A2", "chunk-2", "doc"),
    ]


def make_runner(handler, top_k=5):
    client = httpx.Client(
        base_url="http://test",
        transport=httpx.MockTransport(handler),
    )
    return ShadowRunner(
        base_url="http://test",
        golden_queries=make_queries(),
        top_k=top_k,
        http_client=client,
    )


def ok_handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    return httpx.Response(200, json={
        "question": body["question"],
        "results": [
            {"chunk_id": "chunk-9", "text": "x", "source": "doc",
             "chunk_index": 0, "score": 0.9},
            {"chunk_id": "chunk-1", "text": "y", "source": "doc",
             "chunk_index": 1, "score": 0.8},
        ],
    })


def test_run_once_returns_one_result_per_query():
    results = make_runner(ok_handler).run_once()
    assert len(results) == 2


def test_results_share_one_run_id_and_timestamp():
    results = make_runner(ok_handler).run_once()
    assert len({r.run_id for r in results}) == 1
    assert len({r.timestamp for r in results}) == 1


def test_retrieved_ids_preserve_rank_order():
    result = make_runner(ok_handler).run_once()[0]
    assert result.retrieved_chunk_ids == ["chunk-9", "chunk-1"]
    assert result.scores == [0.9, 0.8]
    assert result.error is None


def test_correct_chunk_id_is_carried_through():
    results = make_runner(ok_handler).run_once()
    assert [r.correct_chunk_id for r in results] == ["chunk-1", "chunk-2"]


def test_sends_question_and_top_k():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return ok_handler(request)

    make_runner(handler, top_k=3).run_once()
    assert seen[0] == {"question": "Question one?", "top_k": 3}


def test_server_error_is_recorded_not_raised():
    def handler(request):
        return httpx.Response(500)

    results = make_runner(handler).run_once()
    assert len(results) == 2  # run completed despite failures
    for r in results:
        assert r.error is not None
        assert r.retrieved_chunk_ids == []


def test_one_failure_does_not_stop_other_queries():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(500)
        return ok_handler(request)

    results = make_runner(handler).run_once()
    assert results[0].error is not None
    assert results[1].error is None
