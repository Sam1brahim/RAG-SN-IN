from langfuse import get_client

langfuse = get_client()

_enabled = None
from langfuse import get_client

langfuse = get_client()

# Verify connection
if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
    _enabled = True
else:
    print("Authentication failed. Please check your credentials and host.")
