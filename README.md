# Multi-Omics VAE Literature RAG Assistant

An AWS-based LLM literature assistant for exploring scientific papers and PhD theses about Variational Autoencoders (VAEs) in single-cell and multi-omics research, especially gene expression and chromatin accessibility.

This project was built to make literature review easier. Instead of manually searching through many long technical papers, the system allows the user to ask questions over a private collection of PDF documents and receive grounded answers based on the retrieved paper content.

## Project Overview

Multi-omics VAE literature is difficult to review because papers often use different terminology, datasets, model architectures, latent-space assumptions, and evaluation metrics. This project builds a Retrieval-Augmented Generation pipeline that processes a collection of research PDFs and enables semantic question answering over them.

The pipeline starts from PDF papers stored in Amazon S3, extracts their text, cleans the extracted content, splits the documents into chunks, creates embeddings for semantic search, retrieves the most relevant chunks for a user question, and finally uses an LLM through Amazon Bedrock to generate an answer grounded in the retrieved sources.

## Main Use Case

The assistant is designed for questions such as:

```text
What is the main idea of MultiVI?

Which papers use VAEs for RNA and ATAC integration?

What are the differences between scVAE, MultiVI, Cobolt, and BindVAE?

Which methods discuss latent spaces for gene expression and chromatin accessibility?

Which papers are useful for understanding VAE-based multi-omics integration?

What are common limitations of VAE-based single-cell methods?

Which methods are more related to cross-modal translation between RNA and ATAC?
```

## Technologies Used

```text
Python
AWS SageMaker
Amazon S3
Amazon Bedrock
Amazon Nova
pypdf
sentence-transformers
NumPy
boto3
Git / GitHub
JupyterLab
```

## AWS Services Used

### Amazon SageMaker

SageMaker is used as the cloud-based development environment for running notebooks, writing Python modules, testing the pipeline, and managing the project workflow.

### Amazon S3

Amazon S3 is used as the main storage layer for the project. It stores the raw PDF papers, extracted text, cleaned text, chunks, embeddings, metadata, and generated outputs.

### Amazon Bedrock

Amazon Bedrock is used for the final LLM answering step. After relevant chunks are retrieved, they are sent to a Bedrock-hosted LLM, such as Amazon Nova, to generate a grounded answer.

## Project Pipeline

```text
PDF papers in S3
        ↓
Text extraction with pypdf
        ↓
Light text cleaning
        ↓
Chunking with metadata
        ↓
Embedding generation
        ↓
Semantic retrieval
        ↓
LLM answer generation with Amazon Bedrock
```

## S3 Structure

```text
s3://multiomic-vae-literature-rag-123223178042-eu-north-1-an/

├── papers/
│   ├── raw/
│   ├── text/
│   └── clean_text/
│
├── chunks/
│   └── paper_chunks.jsonl
│
├── embeddings/
│   ├── chunk_embeddings.npy
│   └── chunk_metadata.jsonl
│
├── vector_index/
│
└── outputs/
    └── generated_answers/
```

## Repository Structure

```text
multiomics-vae-lit-assistant/

├── notebooks/
│   ├── 01_pdf_extraction.ipynb
│   ├── 02_text_preprocessing.ipynb
│   ├── 03_chunk_texts.ipynb
│   ├── 04_create_embeddings.ipynb
│   ├── 05_retrieval.ipynb
│   └── 06_rag_answering.ipynb
│
├── src/
│   ├── pdf_loader.py
│   ├── text_cleaning.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── retrieval.py
│   ├── llm_client.py
│   └── rag_pipeline.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Main Components

### 1. PDF Text Extraction

The raw papers are stored in S3 under:

```text
papers/raw/
```

The extraction step reads the PDF files using `pypdf`, extracts embedded text from each document, and saves the extracted text files to:

```text
papers/text/
```

This avoids paid OCR services and works well for digital academic PDFs that already contain selectable text.

### 2. Text Cleaning

The cleaning step performs light preprocessing on extracted text. It removes unnecessary page markers, reduces excessive spaces, fixes broken line breaks, and handles simple hyphenated word splits.

The goal is not to perfectly reconstruct the PDF layout. Tables and formulas may still be imperfect, but the main explanatory text remains usable for retrieval and question answering.

Cleaned files are saved to:

```text
papers/clean_text/
```

### 3. Chunking

Long papers are split into overlapping text chunks. Each chunk keeps metadata, including:

```text
source file
paper name
chunk ID
start word
end word
chunk text
```

The chunks are saved as a JSONL file:

```text
chunks/paper_chunks.jsonl
```

JSONL is used because each line represents one independent chunk with its metadata.

### 4. Embedding Generation

Each chunk is converted into a numerical vector using a sentence-transformer embedding model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The embeddings and metadata are saved to S3:

```text
embeddings/chunk_embeddings.npy
embeddings/chunk_metadata.jsonl
```

The embedding file stores the numerical vectors, while the metadata file stores the corresponding paper names, chunk IDs, and text.

### 5. Semantic Retrieval

When the user asks a question, the question is converted into an embedding using the same embedding model. The system compares the question embedding with all chunk embeddings using cosine similarity and retrieves the top relevant chunks.

This step allows the system to find relevant passages even when the question does not exactly match the wording used in the papers.

### 6. RAG Answer Generation

The retrieved chunks are combined into a prompt and sent to an LLM through Amazon Bedrock. The LLM is instructed to answer using only the retrieved context and cite the source paper names when possible.

This makes the final answer more grounded than a normal chatbot response.

## Example Workflow

```text
User question:
"What is the main idea of BindVAE?"

System:
1. Embeds the question
2. Retrieves relevant chunks from the paper collection
3. Builds a RAG prompt using those chunks
4. Sends the prompt to Amazon Bedrock
5. Returns an answer grounded in the retrieved paper content
```

## Why This Project Is Useful

This project helps with literature review in a technical research area where papers are long, dense, and difficult to compare manually.

It can help with:

```text
finding relevant papers
summarizing method ideas
comparing VAE-based models
understanding latent-space designs
identifying datasets and evaluation strategies
finding limitations and research gaps
organizing thesis-related literature
```

## Current Status

Completed:

```text
AWS project setup
GitHub repository setup
S3 project structure
PDF paper storage in S3
PDF text extraction
text cleaning utility
chunk generation
embedding generation
semantic retrieval
```

In progress:

```text
Amazon Bedrock answer generation
RAG prompt refinement
README/documentation polish
```

## Limitations

PDF extraction quality depends on the structure of each PDF. Tables, formulas, footers, and multi-column layouts may not always be extracted perfectly.

The system currently uses lightweight text cleaning rather than advanced PDF layout reconstruction.

The retrieval quality depends on the embedding model, chunk size, chunk overlap, and the quality of extracted text.

This project is designed as a literature review assistant, not as a replacement for carefully reading the original papers.

## Future Improvements

Possible improvements include:

```text
adding a Streamlit interface
adding page-level source references
improving paper metadata extraction
generating paper-level summaries
creating comparison tables across methods
using Bedrock embedding models
automating PDF ingestion with AWS Lambda
deploying an API with API Gateway
adding Docker support
adding evaluation questions for retrieval quality
```

## Project Goal

The goal of this project is to build a practical end-to-end LLM system for scientific literature review. It combines cloud storage, document processing, semantic search, retrieval-augmented generation, and AWS-based LLM integration in a research domain related to multi-omics VAEs.
