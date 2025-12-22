# =============================================================================
# Multi-stage Dockerfile for ML Model Inference
# =============================================================================
# Stage 1: Builder - Install dependencies
# =============================================================================
FROM python:3.11-slim as builder

WORKDIR /build

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies to a virtual environment
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --target=/install \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    mlflow==2.10.0 \
    pandas==2.1.3 \
    scikit-learn==1.3.2 \
    xgboost==2.0.3 \
    pydantic==2.5.0 \
    python-multipart==0.0.6

# =============================================================================
# Stage 2: Runtime - Minimal production image
# =============================================================================
FROM python:3.11-slim

WORKDIR /app

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local/lib/python3.11/site-packages

# Copy application code
COPY src/ ./src/
COPY models/ ./models/

# Build arguments
ARG MODEL_VERSION=latest
ARG MLFLOW_TRACKING_URI=http://127.0.0.1:5001
ARG MODEL_NAME=iris-classifier

# Environment variables
ENV MODEL_VERSION=${MODEL_VERSION} \
    MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} \
    MODEL_NAME=${MODEL_NAME} \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Create non-root user
RUN useradd -m -u 1000 mlops && \
    chown -R mlops:mlops /app

USER mlops

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose port
EXPOSE ${PORT}

# Run the application
CMD ["sh", "-c", "uvicorn src.serve:app --host 0.0.0.0 --port ${PORT} --workers 4"]
