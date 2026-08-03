# retrieval.py

from .search import dense_search, sparse_search


def retrieve_chunks(question, top_k=5):
    dense_results = dense_search(
        question,
        top_k=20
    )

    sparse_results = sparse_search(
        question,
        top_k=20
    )

    combined_results = combine_results(
        dense_results,
        sparse_results
    )

    final_results = rerank_results(
        question,
        combined_results
    )

    return final_results[:top_k]