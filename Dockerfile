FROM python:3.12-slim

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Default command (for API server)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]# Dockerfile for JusticeLens AI
