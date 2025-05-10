# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies, including libmagic
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    libmagic-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire app into container
COPY . .

# Expose port
EXPOSE 8000

# Run the server
CMD ["uvicorn", "server.websocket_server:app", "--host", "0.0.0.0", "--port", "8000"]
