from typing import List, Dict
from pathlib import Path


def split_text_into_chunks(
    text: str,
    source_file: str,
    chunk_size_words: int = 900,
    overlap_words: int = 150,
) -> List[Dict]:
    """
    Split one paper text into overlapping word-based chunks.

    Each chunk keeps metadata so later RAG can cite the source paper.
    """

    words = text.split()

    chunks = []
    start = 0
    chunk_id = 0

    while start < len(words):
        end = start + chunk_size_words
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        chunks.append(
            {
                "source_file": source_file,
                "paper_name": Path(source_file).stem,
                "chunk_id": chunk_id,
                "start_word": start,
                "end_word": min(end, len(words)),
                "text": chunk_text,
            }
        )

        chunk_id += 1
        start += chunk_size_words - overlap_words

    return chunks