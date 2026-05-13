from typing import List, Dict

from retrieval import retrieve_top_k_chunks


def build_rag_prompt(question: str, retrieved_chunks: List[Dict]) -> str:
    context_blocks = []

    for chunk in retrieved_chunks:
        block = f"""
Source paper: {chunk["paper_name"]}
Chunk ID: {chunk["chunk_id"]}

{chunk["text"]}
"""
        context_blocks.append(block)

    context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""
You are a research assistant helping with a literature review on multi-omics VAE papers.

Answer the question using ONLY the provided context.
If the answer is not in the context, say that the provided papers do not contain enough information.
Cite the source paper names in your answer.

Question:
{question}

Context:
{context}

Answer:
"""

    return prompt.strip()


def answer_question_with_rag(
    question: str,
    embedding_model,
    llm_client,
    chunk_embeddings,
    chunk_metadata,
    top_k: int = 5,
) -> Dict:
    retrieved_chunks = retrieve_top_k_chunks(
        question=question,
        model=embedding_model,
        chunk_embeddings=chunk_embeddings,
        chunk_metadata=chunk_metadata,
        top_k=top_k,
    )

    prompt = build_rag_prompt(
        question=question,
        retrieved_chunks=retrieved_chunks,
    )

    answer = llm_client.generate_answer(prompt)

    return {
        "question": question,
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
    }