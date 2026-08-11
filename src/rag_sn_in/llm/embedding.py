from sentence_transformers import SentenceTransformer

from rag_sn_in.config import EMBEDDING_MODEL_NAME
"""
Module specialises in building embeddings only, following needed embedding models.
Generally it receives one prompt or text at a time, for conversion
"""

# Single source of truth: rag_sn_in.config.EMBEDDING_MODEL_NAME (the same
# model whose tokenizer sizes the chunks in processing/chunking.py).
MODEL_NAME = EMBEDDING_MODEL_NAME
_model = SentenceTransformer(MODEL_NAME,local_files_only=True)

# embed_document(chunk["text"]) for stored chunks
# 1. FOR DOCUMENTS: Do NOT use the query prefix.
def embed_document(text: str):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")

    return _model.encode(
        text,
        convert_to_numpy=True,
        prompt_name="document"
    )

# 2. FOR QUERIES: Explicitly use the query prefix.
def embed_query(text: str):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")

    return _model.encode(
        text,
        convert_to_numpy=True,
        prompt_name="query" # <--- Explicitly tell it this is a query
    )

def embed_queries_batch(texts: list[str]):
    if not texts:
        return []
    return _model.encode(
        texts,
        convert_to_numpy=True,
        batch_size=32,
        prompt_name="query" # <--- Explicitly tell it this is a query
    )

def unload_embedder():
    """Module-level _model survives gc.collect(); null it before loading
    another large model on the same GPU."""
    global _model
    _model = None
