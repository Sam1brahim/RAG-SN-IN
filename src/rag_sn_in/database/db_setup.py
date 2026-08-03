from qdrant_client.models import VectorParams, Distance
from rag_sn_in.database.client import get_client


def ensure_collection(client, collection_name, vector_size, distance=Distance.COSINE):
    if client.collection_exists(collection_name):
        print(f"Collection '{collection_name}' already exists. Skipping creation.")
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=distance
        )
    )
    print(f"Collection '{collection_name}' created with vector size {vector_size} and distance {distance}.")


if __name__ == "__main__":
    client = get_client()
    ensure_collection(client, collection_name="knowledge_base", vector_size=768)