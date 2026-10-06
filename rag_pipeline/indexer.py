import uuid
import weaviate
from rag_pipeline.embedder import Embedder

CLASS_NAME = "DocumentChunk"

def get_weaviate_client(url: str = "http://localhost:8000") -> weaviate.Client:
    """
    Returns a connected weaviate client
    Separated from the Indexer class so it can be reused by query layer
    without needing to import the entire indexer
    """
    return weaviate.Client(url)

class Indexer:
    """
    Takes raw documents, splits them into chunks, embeds each chunks,
    and writes them into Weaviate

    Chunking strategy: fixed-size with overlap.
    - chunk_size: number of chars per chunk
    - chunk_overlap: num of chars shared by two adjacent chunks
        to preserve boundary and context
    """
    def __init__(self,
                 embedder: Embedder,
                 client: weaviate.Client,
                 chunk_size: int = 512,
                 chunk_overlap: int = 50):
        self.embedder = embedder
        self.client = client
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap


    def setup_schema(self) -> None:
        """
        Creates the DocumentChunk class in Weaviate if it doesn't exist
        Safe to call multiple times

        We define schema explicitly as a safe production practice even
        though Weaviate can auto-schema.
        :return:
        """
        if self.client.schema.exists(CLASS_NAME):
            return

        schema = {
            "class": CLASS_NAME,
            "vectorized": "none",
            "properties": [
                {
                    "name": "chunk_id",
                    "dataType": ["text"],
                },
                {
                    "name": "text",
                    "dataType": ["text"]
                },
                {
                    "name": "source",
                    "dataType": ["text"],
                },
                {
                    "name": "chunk_index",
                    "dataType": ["int"]
                }
            ]
        }
        self.client.schema.create_class(schema)


    def _chunk_text(self, text: str) -> list[str]:
        """
        Splits text into overlapping fixed-size chunks.

        Example with chunk_size=20, chunk_overlap=5:
        "abcdefghijklmnopqrstuvwxyz"
        → ["abcdefghijklmnopqrst", "pqrstuvwxyz"]
                                    ↑ 5 char overlap

        The step size is (chunk_size - chunk_overlap), so each
        new chunk starts chunk_overlap characters before the
        previous chunk ended.
        """
        chunks = []
        step = self.chunk_size - self.chunk_overlap
        for i in range(0, len(text), step):
            chunk = text[i: i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk)

        return chunks

    def index_document(self, text: str, source: str) -> list[str]:
        """
        Chunks, embeds, and indexes a single document.

        Returns a list of chunk_ids that were created - important for golden query
        set as we need to know what which chunk_id contains the answer to each golden
        query.
        """
        chunks = self._chunk_text(text)
        vectors = self.embedder.embed_batch(chunks)
        chunk_ids = []

        with self.client.batch as batch:
            batch.batch_size = 50   # we write in batches of 50 for efficiency

            for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
                chunk_id = str(uuid.uuid4())

                batch.add_object(
                    class_name = CLASS_NAME,
                    data_object = {
                        "chunk_id": chunk_id,
                        "text": chunk,
                        "source": source,
                        "chunk_index": i,
                    },
                    vector = vector.tolist()
                )
                chunk_ids.append(chunk_id)

        return chunk_ids

    def clear_index(self) -> None:
        """
        Deletes and recreates the schema — wipes all indexed data.
        Used by the drift injection script to simulate corpus changes.
        Only call this in development/testing.
        """
        if self.client.schema.exists(CLASS_NAME):
            self.client.schema.delete_class(CLASS_NAME)
        self.setup_schema()
