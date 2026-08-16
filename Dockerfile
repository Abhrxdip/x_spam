# Python 3.12 slim base
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir shap lime gunicorn

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

