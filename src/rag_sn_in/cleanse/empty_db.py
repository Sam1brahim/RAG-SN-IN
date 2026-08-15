# cleanup any collection from the DB
from rag_sn_in.database.client import get_client

def collection_exists(client, collection_name: str) -> bool:
    return client.collection_exists(collection_name)

if __name__ == '__main__':
    collection_name = "railway"
    client = get_client()
    if collection_exists(client,collection_name):
        client.delete_collection(collection_name)
        print(f"{collection_name} collection deleted.")
        client.close()
    else:
        print("No such Collection Exists in Qdrant DB")
        client.close()