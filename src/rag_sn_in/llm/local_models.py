"""
Local models via Ollama + LangChain — learning starter for RAG-SN-IN.

Two different jobs, two different models:

1) EMBEDDING model  -> turns text into vectors (retrieval / search)
2) CHAT / LLM model -> reads retrieved chunks and writes an answer

Your chunking already targets EmbeddingGemma token limits
(google/embeddinggemma-300m). Serving the same family via Ollama
keeps indexing and search consistent.

---------------------------------------------------------------
One-time setup (run in a terminal, outside Python):

  # chat model (pick a size your GPU/RAM can hold)
  ollama pull gemma4:12b
  # or: gemma4:e4b  (lighter) / gemma4:26b (heavier)

  # same embedding family you designed chunks for
  ollama pull embeddinggemma

  # Python deps (from project root, with your env active)
  uv add langchain-ollama langchain-core

---------------------------------------------------------------
Sanity checks:

  ollama list
  ollama run gemma4:12b "Dis en une phrase ce qu'est le DRR."

Then run this file:

  python -m rag_sn_in.llm.local_models
"""

from __future__ import annotations

from langchain_ollama import ChatOllama, OllamaEmbeddings


# ---- config: change tags here, nowhere else ----
CHAT_MODEL = "gemma4:e4b"          # generation (answers)
EMBED_MODEL = "embeddinggemma"     # vectors (retrieval)
OLLAMA_BASE_URL = "http://localhost:11434"


def make_chat_llm() -> ChatOllama:
    """
    Chat LLM = the 'brain' that writes answers.

    temperature=0 -> more deterministic (better for eval / legal text).
    """
    return ChatOllama(
        model=CHAT_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )


def make_embedder() -> OllamaEmbeddings:
    """
    Embeddings = the 'search engine' encoder.

    Same text -> same vector (approximately). Similarity between
    query vector and chunk vectors is how RAG finds candidates.
    """
    return OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def demo_chat() -> None:
    llm = make_chat_llm()
    # .invoke takes a string or a list of messages
    response = llm.invoke(
        "En une phrase: quel est l'objectif du Document de "
        "Référence du Réseau (DRR) de SNCF Réseau ?"
    )
    # response.content is the text; response may also carry metadata
    print("=== CHAT ===")
    print(response.content)


def demo_embed() -> None:
    embedder = make_embedder()

    query = "Comment demander une acceptation de non-conformité ?"
    chunk = (
        "Dès lors que l'EF détecte avant départ une possible "
        "dégradation de la performance du sillon, elle doit "
        "demander l'acceptation via le processus DANC."
    )
    unrelated = "Recette de tarte aux pommes."

    # embed_query: one string (the user question)
    q_vec = embedder.embed_query(query)
    # embed_documents: list of strings (your chunks)
    doc_vecs = embedder.embed_documents([chunk, unrelated])

    def cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        return dot / (na * nb)

    print("=== EMBED ===")
    print(f"vector dim: {len(q_vec)}")
    print(f"sim(query, relevant chunk):   {cosine(q_vec, doc_vecs[0]):.4f}")
    print(f"sim(query, unrelated text):   {cosine(q_vec, doc_vecs[1]):.4f}")
    print("(relevant should score higher — that is retrieval)")


if __name__ == "__main__":
    # Comment one out if the model is not pulled yet.
    demo_embed()
    demo_chat()
