import atexit
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


def close_client() -> None:
    """Close the embedded Qdrant store before interpreter shutdown.

    QdrantClient.__del__ tries to import modules after sys.meta_path is
    already None, which prints an ignored exception even when the run succeeded.
    """
    global _client
    if _client is None:
        return
    try:
        _client.close()
    except Exception:
        pass
    _client = None


atexit.register(close_client)
