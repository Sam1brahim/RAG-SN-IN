# search.py
from rag_sn_in.database.client import get_client
from rag_sn_in.llm.embedding import embed_query

def dense_search(client, query, collection_name, top_k=10):
    query_vector = embed_query(query)
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k
    )
    return results

if __name__ == "__main__":
    client = get_client()
    collection_name = "DRR_SNCF"
    query = "Selon le Document de référence du réseau 2025, quelles règles s'appliquent pour : Responsabilités particulières des demandeurs autres qu'entreprise ferroviaire ?"
    results = dense_search(client, query, collection_name, top_k=10)
    for r in results:
        print('\n',r,'\n')


#def sparse_search(query,embed_query, collection_name,top_k=10):
   # """
   # Search the BM25 or keyword index.
   # """
   # # 1. Tokenize query
    # 2. Search BM25 index
   # # 3. Return results
   # pass