from pathlib import Path
from qdrant_client import QdrantClient

project_root = Path(__file__).resolve().parent.parent.parent.parent
db_path = project_root / "data" / "vector_db" / "qdrant"

_client = None

def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(path=str(db_path))
    return _client