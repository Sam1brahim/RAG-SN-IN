import json
from qdrant_client.models import PointStruct
import uuid

"""
Never gets called for a query, this is only for storing Chunks inside the VECTOR DB: Qdrant in my case.
"""
def index_chunks(chunks_path, client,embed_document,collection_name):
    vectorised_chunks = []
    with open(chunks_path, "r") as f:
        for line in f:
            chunk = json.loads(line)
            vectorised = embed_document(chunk['text'])
            vectorised_chunks.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vectorised,
                        payload={
                            "id": chunk["id"],
                            "text": chunk["text"],
                            "document_id": chunk["document_id"],
                            "page_start": chunk["page_start"],
                            "page_end": chunk["page_end"],
                            "token_count": chunk["token_count"],
                            "section_path": chunk["section_path"],
                        }
                    )
                )
    client.upsert(
        collection_name=collection_name,
        points=vectorised_chunks
    )
    return print('Number of Vectors Persisted:', len(vectorised_chunks))