"""
Shared project configuration.

Single source of truth for the embedding model name and data paths, so
chunking (token counting), embedding, indexing and eval tooling never
drift apart.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Used BOTH for token counting (processing/chunking.py) and for embedding
# (llm/embedding.py). Token counts are only meaningful in the units of the
# model that will consume the chunks.
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

DATA_PROCESSED_TEXT_DIR = ROOT / "data" / "processed" / "text"
DATA_CHUNKS_DIR = ROOT / "data" / "processed" / "chunks"
