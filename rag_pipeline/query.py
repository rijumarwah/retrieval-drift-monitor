import weaviate
from rag_pipeline.embedder import Embedder
from rag_pipeline.indexer import CLASS_NAME

class QueryEngine:
    """
    Accepts a natural language question, embeds it, and returns the top-k most semantically
    similar chunks from Weaviate.
    """

    def __init__(self, embedder: Embedder, client: weaviate.Client):
        self.embedder = embedder
        self.client = client

    def query(self, question: str, top_k: int = 5) -> list[dict]:
        """
        Embeds the question and retrieves top-k chunks from Weaviate.

        Returns a list of dicts, each containing:
          - chunk_id:    the unique ID of the chunk
          - text:        the chunk text
          - source:      the document it came from
          - chunk_index: position of this chunk in the original document
          - score:       the cosine similarity score (higher = more similar)

        Why return score?
        The monitoring layer needs it to track cosine similarity drift
        over time — not just whether the right chunk was returned,
        but how similar it was.
        """
        vector = self.embedder.embed(question).tolist()

        result = (
            self.client.query
            .get(CLASS_NAME, ["chunk_id", "text", "source", "chunk_index"])
            .with_near_vector({"vector": vector})
            .with_limit(top_k)
            .with_additional(["certainty"])     # certainty = cosine similarity in Weaviate
            .do()
        )

        hits = result.get("data", {}).get("Get", {}).get(CLASS_NAME , [])

        return [
            {
                "chunk_id": hit["chunk_id"],
                "text": hit["text"],
                "source": hit["source"],
                "chunk_index": hit["chunk_index"],
                "score": hit.get("_additional", {}).get("certainty", 0.0),
            }
            for hit in hits
        ]
