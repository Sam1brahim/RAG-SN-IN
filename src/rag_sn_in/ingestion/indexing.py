import json
from qdrant_client.models import PointStruct
import uuid
from pathlib import Path
from rag_sn_in.database.client import get_client
from rag_sn_in.llm.embedding import embed_document
from rag_sn_in.database.db_setup import ensure_ToCreate_collection
from rag_sn_in.config import VECTOR_SIZE

"""
Never gets called for a query, this is only for storing Chunks inside the VECTOR DB: Qdrant in my case.
"""
def index_chunks(chunks_path, client, embed_document, collection_name, batch_size=100):
    vectorised_chunks = []
    total_persisted = 0
    print('Indexing Your Chunks. . .')
    chunks_dir = Path(chunks_path)
    jsonl_files = sorted(chunks_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No .jsonl files found in {chunks_dir}")
        return 0
    for file_path in jsonl_files:
        print(f'Processing file: {file_path.name}')
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                chunk = json.loads(line)
                vectorised = embed_document(chunk['text'])
                vectorised_chunks.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vectorised,
                        payload={
                            "id": chunk["chunk_id"],
                            "text": chunk["text"],
                            "document_id": chunk["document_id"],
                            "page_start": chunk["page_start"],
                            "page_end": chunk["page_end"],
                            "token_count": chunk["token_count"],
                            "section_path": chunk["section_path"],
                        }
                    )
                )

                if len(vectorised_chunks) >= batch_size:
                    client.upsert(
                        collection_name=collection_name,
                        points=vectorised_chunks
                    )
                    total_persisted += len(vectorised_chunks)
                    print(f'Batch saved — {total_persisted} chunks persisted so far.')
                    vectorised_chunks = []

        # flush remaining chunks that didn't fill a full batch
        if vectorised_chunks:
            client.upsert(
                collection_name=collection_name,
                points=vectorised_chunks
            )
            total_persisted += len(vectorised_chunks)
            print(f'Final batch saved — {total_persisted} chunks persisted total.')

    print('Number of Vectors Persisted:', total_persisted)
    return total_persisted

if __name__ == '__main__':
    collection_name = 'DRR_SNCF'
    client = get_client()
    ensure_ToCreate_collection(client,collection_name=collection_name,vector_size=VECTOR_SIZE) # to auto check if it exists, if not, build it inside client
    index_chunks(r"E:\Project RAG-SN-IN\data\processed\chunks", client,embed_document,collection_name)