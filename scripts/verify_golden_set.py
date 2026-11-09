"""
Checks that each golden query's expected answer actually appears
in its correct chunk. Paraphrased answers (e.g. q005) will show
NO and need a manual read; short factual answers should show YES.

    python -m scripts.verify_golden_set
"""
from eval.golden_set_loader import load_golden_set
from rag_pipeline.indexer import get_weaviate_client, CLASS_NAME


def main():
    client = get_weaviate_client()

    result = (
        client.query
        .get(CLASS_NAME, ["chunk_id", "text"])
        .with_limit(5000)
        .do()
    )
    chunks = {
        h["chunk_id"]: h["text"]
        for h in result["data"]["Get"][CLASS_NAME]
    }
    print(f"Fetched {len(chunks)} chunks from Weaviate\n")

    for q in load_golden_set():
        text = chunks.get(q.correct_chunk_id)
        if text is None:
            print(f"{q.query_id}: CHUNK NOT FOUND")
            continue
        found = q.expected_answer.lower() in text.lower()
        print(f"{q.query_id}: {'YES' if found else 'NO '}  "
              f"expected='{q.expected_answer[:40]}'")


if __name__ == "__main__":
    main()
