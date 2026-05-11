from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


def load_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Load a sentence-transformers embedding model.
    """
    model = SentenceTransformer(model_name)
    return model


def create_embeddings(
    texts: List[str],
    model,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Convert a list of text chunks into embedding vectors.
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings