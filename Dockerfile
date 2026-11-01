FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml requirements.txt requirements-demucs.txt ./
COPY src ./src
COPY backend ./backend
COPY scripts ./scripts

RUN pip install --no-cache-dir .[roformer]

ENV PYTHONPATH=/app/src:/app
ENV ATT_DATA_DIR=/app/data

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
