FROM python:3.10-slim

# System dependencies for chromadb, numpy, etc.
RUN apt-get update && apt-get install -y \
    git build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and data
COPY . .

# Pre-download embedding model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Preload chroma_db at build time (optional, or let it happen on first run)

CMD ["python", "src/recommender.py"]
