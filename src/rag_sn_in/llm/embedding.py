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

# Constants for explicit prompts (safer than relying on prompt_name mapping)
QUERY_PROMPT = "task: search result | query: "
DOC_PROMPT = "title: none | text: "

# embed_document(chunk["text"]) for stored chunks
# 1. FOR DOCUMENTS: Do NOT use the query prefix.
def embed_document(text: str):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")

    # Explicitly prepend the prompt as recommended by Google for Gemma-300m
    return _model.encode(
        DOC_PROMPT + text,
        convert_to_numpy=True
    )

# 2. FOR QUERIES: Explicitly use the query prefix.
def embed_query(text: str):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")

    # Explicitly prepend the prompt as recommended by Google for Gemma-300m
    return _model.encode(
        QUERY_PROMPT + text,
        convert_to_numpy=True
    )

def embed_queries_batch(texts: list[str]):
    if not texts:
        return []
    
    # Prepend prompt to each query
    prefixed_texts = [QUERY_PROMPT + t for t in texts]
    
    return _model.encode(
        prefixed_texts,
        convert_to_numpy=True,
        batch_size=32
    )

def unload_embedder():
    """Module-level _model survives gc.collect(); null it before loading
    another large model on the same GPU."""
    global _model
    _model = None
