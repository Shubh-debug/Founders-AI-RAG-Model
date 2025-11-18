# ===============================
# Base Image
# ===============================
FROM python:3.13-slim

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside the container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
	pip install --no-cache-dir -r requirements.txt

# Copy the application package to preserve package path `app.*`
COPY app ./app

# Expose port for FastAPI
EXPOSE 8000

# Default command to run FastAPI (respect Railway $PORT)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
