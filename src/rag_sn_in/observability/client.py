from langfuse import Langfuse
import os

from dotenv import load_dotenv
load_dotenv()

_client = None

def get_client():
    """Lazy initialization of Langfuse client to ensure .env is loaded first."""
    global _client
    if _client is None:
        _client = Langfuse()
        try:
            if not _client.auth_check():
                print("Langfuse authentication failed. Check credentials/host.")
        except Exception as e:
            print(f"Langfuse auth check error: {e}")
    return _client

def is_enabled():
    """Checks if Langfuse is configured and reachable."""
    client = get_client()
    try:
        return client.auth_check()
    except:
        return False
