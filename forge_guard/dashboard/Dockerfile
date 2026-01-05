FROM python:3.12-slim

# Create non-root user
RUN addgroup --gid 1000 app && \
    adduser --uid 1000 --gid 1000 --disabled-password --gecos "" app

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

ENV PYTHONUNBUFFERED=1

USER app

CMD ["streamlit", "run", "forge_guard/dashboard/app.py", "--server.port", "8501", "--server.headless", "true"]
