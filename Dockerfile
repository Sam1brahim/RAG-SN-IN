FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for PyMuPDF and Docling
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN pip install uv

# Copy dependency files first (layer caching)
COPY pyproject.toml uv.lock ./

# Install Python dependencies
RUN uv sync --no-dev

# Copy source code
COPY src/ ./src/
COPY start.py ./

# Create data directories
RUN mkdir -p data/raw data/processed data/eval data/ragas data/vector_db/qdrant

EXPOSE 8000

# Default: launch the interactive CLI
# Override with: docker run ... uvicorn rag_sn_in.api.main:app --host 0.0.0.0 --port 8000
CMD ["python", "start.py"]
