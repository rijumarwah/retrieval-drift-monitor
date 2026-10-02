import numpy
from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"


class Embedder:
    """
    The embedder wraps a sentence-transformer model to produce fixed-size vectors
    from raw text input.

    Model used: all-MiniLM-L6-v2
    - 384 dimension output vectors
    - Fast inference, runs on CPU
    - Strong performance
    - No API key required: loaded and ran locally
    """
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> np.ndarray:
        """
        Embed a single string to 1D numpy array of shape (384,)
        :param text: Input string to embed
        :return: 1D numpy array
        """
        return self.model.encode(text, convert_to_numpy=True)

    def embed_batch(self, texts: list[str]) -> numpy.ndarray:
        """
        Embeds a list of strings to 2D numpy array of shape (N, 384)
        More efficient than calling embed() in a loop, the model processes texts
        in parallel internally
        :param texts: Input strings to embed
        :return: 2D numpy array
        """
        return self.model.encode(texts, convert_to_numpy=True)

    @property
    def dimension(self) -> int:
        """
        Gives the embedding dimension. Useful when configuring Weaviate, which
        requires to know what the vector size is beforehand.
        :return: Embedding dimension
        """
        return self.model.get_embedding_dimension()
