"""
Shared project configuration.

Single source of truth for the embedding model name and data paths, so
chunking (token counting), embedding, indexing and eval tooling never
drift apart.

All paths are derived from the project root — no hardcoded absolute paths.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

EMBEDDING_MODEL_NAME = "google/embeddinggemma-300m"

VECTOR_SIZE = 768

RERANKER_NAME = "BAAI/bge-reranker-v2-m3"

LLM_NAME = "gemma4-e2b"
EVALUATOR = "deepseek-chat"

DATA_PROCESSED_TEXT_DIR = ROOT / "data" / "processed" / "text"
DATA_CHUNKS_DIR = ROOT / "data" / "processed" / "chunks"
DATA_CHUNKS_512_DIR = DATA_CHUNKS_DIR / "max token 512"
DATA_EVAL_DIR = ROOT / "data" / "eval" / "eval chunks 512 max"
DATA_RAGAS_DIR = ROOT / "data" / "ragas"
DATA_EVAL_METRICS_DIR = ROOT / "data" / "Eval RAG metrics" / "ragas"
DATA_RAW_DIR = ROOT / "data" / "raw"
