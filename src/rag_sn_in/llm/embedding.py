from sentence_transformers import SentenceTransformer
"""
Module specialises in building embeddings only, following needed embedding models.
Generally it receives one prompt or text at a time, for conversion
"""

MODEL_NAME = "google/embeddinggemma-300m"

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
