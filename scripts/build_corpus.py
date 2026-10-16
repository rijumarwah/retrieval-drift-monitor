"""
Fetches Wikipedia articles, indexes them into Weaviate,
and prints the chunk_ids so we can build the golden query set.

Run this once to populate Weaviate:
    python scripts/build_corpus.py
"""
import json
import wikipedia
from rag_pipeline.embedder import Embedder
from rag_pipeline.indexer import Indexer, get_weaviate_client

ARTICLES = [
    "Python (programming language)",
    "Machine learning",
    "Paris",
    "Solar System",
    "World War II",
]

def main():
    print("Connecting to Weaviate...")
    client = get_weaviate_client()

    print("Loading embedding model...")
    embedder = Embedder()

    indexer = Indexer(embedder=embedder, client=client)

    print("Setting up Weaviate schema")
    indexer.setup_schema()

    corpus_map = {}     # article title → list of chunk_ids

    for title in ARTICLES:
        print(f"Fetching: {title}")
        try:
            page = wikipedia.page(title, auto_suggest=False)
            text = page.content
        except Exception as ex:
            print(f"Skipping {title}: {ex}")
            continue

        print(f"Indexing {len(text)} characters...")
        chunk_ids = indexer.index_document(text=text, source=title)
        corpus_map[title] = chunk_ids
        print(f"  Created {len(chunk_ids)} chunks")

    # Save the corpus map so we can look up chunk_ids
    # when building the golden query set
    with open("eval/corpus_map.json", "w") as f:
        json.dump(corpus_map, f, indent=2)

    print("Corpus Map saved to eval/corpus_map.json")
    print("Use the chunk_ids in that file to build eval/golden_set.json")


if __name__ == "__main__":
    main()
