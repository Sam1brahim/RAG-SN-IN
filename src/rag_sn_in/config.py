"""
Shared project configuration.

Single source of truth for the embedding model name and data paths, so
chunking (token counting), embedding, indexing and eval tooling never
drift apart.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

EMBEDDING_MODEL_NAME = "google/embeddinggemma-300m"

VECTOR_SIZE = 768

RERANKER_NAME= "BAAI/bge-reranker-v2-m3"

LLM_NAME= "qwen"

### LLMs to Try:
    # "Qwen/Qwen2.5-7B-Instruct"
    # "google/gemma-4-E4B-it"
    # "google/gemma-4-E2B-it"
    # "google/gemma-2-9b-it"

### Embedding Models:

    # "google/embeddinggemma-300m" 768 
    # "BAAI/bge-m3" 1024
    # "nomic-ai/nomic-embed-text-v1.5" 768 
    # "intfloat/multilingual-e5-large" 1024

### Rerankers:

    # "Qwen/Qwen3-Reranker-0.6B"
    # "BAAI/bge-reranker-v2-m3"

DATA_PROCESSED_TEXT_DIR = ROOT / "data" / "processed" / "text"
DATA_CHUNKS_DIR = ROOT / "data" / "processed" / "chunks"
