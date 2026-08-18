# Use Python 3.11-slim as base image for efficiency
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for PyTorch and image processing
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port 8000
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Start FastAPI via Uvicorn targeting model_service_app.py
CMD ["uvicorn", "src.model_service_app:app", "--host", "0.0.0.0", "--port", "8000"]
