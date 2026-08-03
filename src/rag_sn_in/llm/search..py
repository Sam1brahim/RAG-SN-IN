# search.py


def dense_search(query,embed_query, collection_name,top_k=10):
    """
    Search the vector database using semantic similarity.
    """
    vectorised_chunks = []
    with open(chunks_path, "r") as f:
            for line in f:
                chunk = json.loads(line)
    pass


def sparse_search(query,embed_query, collection_name,top_k=10):
    """
    Search the BM25 or keyword index.
    """
    # 1. Tokenize query
    # 2. Search BM25 index
    # 3. Return results
    pass