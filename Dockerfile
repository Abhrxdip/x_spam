# Python 3.11 slim base (optimal wheel compatibility)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements and install CPU-optimized packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=5000
ENV FLASK_DEBUG=False

# Expose container port
EXPOSE 5000

# Start server using Gunicorn (dynamically binds to Render $PORT)
CMD ["sh", "-c", "gunicorn app:app --timeout 120 --workers 2 --bind 0.0.0.0:${PORT:-5000}"]
