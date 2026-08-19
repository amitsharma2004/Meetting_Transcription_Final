FROM python:3.11-slim

WORKDIR /app

# Set environment variables for Python & PyTorch CPU execution
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_HOME=/app/pretrained_models \
    HF_HOME=/app/pretrained_models

# Install system dependencies:
# - ffmpeg: Audio decoding for Whisper/STT
# - libsndfile1: Required by soundfile / torchaudio / speechbrain
# - libgomp1: OpenMP runtime for CPU tensor ops
# - curl & git: Healthchecks and model downloading
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, and wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and static frontend
COPY . .

# Create data and pretrained_models directories
RUN mkdir -p /app/data /app/pretrained_models

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
