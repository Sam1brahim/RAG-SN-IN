from sentence_transformers import SentenceTransformer

from rag_sn_in.config import EMBEDDING_MODEL_NAME
"""
Module specialises in building embeddings only, following needed embedding models.
Generally it receives one prompt or text at a time, for conversion
"""

# Single source of truth: rag_sn_in.config.EMBEDDING_MODEL_NAME (the same
# model whose tokenizer sizes the chunks in processing/chunking.py).
MODEL_NAME = EMBEDDING_MODEL_NAME
_model = SentenceTransformer(MODEL_NAME)

# embed_document(chunk["text"]) for stored chunks
def embed_document(text: str):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")

    return _model.encode_document(
        text,
        convert_to_numpy=True
    )

# embed_query(user_question) for search questions.
def embed_query(text: str):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")

    return _model.encode_query(
        text,
        convert_to_numpy=True
    )

def embed_queries_batch(texts: list[str]):
    if not texts:
        return []
    # SentenceTransformer natively processes lists much faster in parallel batches
    return _model.encode_query(
        texts,
        convert_to_numpy=True,
        batch_size=32 
    )
