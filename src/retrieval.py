import numpy as np
from typing import List, Dict


def retrieve_top_k_chunks(
    question: str,
    model,
    chunk_embeddings: np.ndarray,
    chunk_metadata: List[Dict],
    top_k: int = 5,
) -> List[Dict]:
    """
    Retrieve the top-k most relevant chunks for a question.

    Because embeddings are normalized, dot product works like cosine similarity.
    """

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    scores = np.dot(chunk_embeddings, question_embedding)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for rank, idx in enumerate(top_indices, start=1):
        chunk = dict(chunk_metadata[idx])
        chunk["rank"] = rank
        chunk["score"] = float(scores[idx])
        results.append(chunk)

    return results