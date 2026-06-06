# Fraud scoring API image. Mirrors the team's bert_service Dockerfile layout.
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps for psycopg / building wheels kept minimal.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install the CPU build of torch first (avoids pulling the large CUDA wheels).
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# EXTRAS lets the streaming consumer reuse this image with the kafka client:
#   docker build --build-arg EXTRAS="[streaming]" ...
ARG EXTRAS=""
# Install dependencies against a minimal package skeleton first, so editing source code
# below doesn't invalidate the (slow) dependency layer — only the COPY src layer rebuilds.
COPY pyproject.toml README.md ./
RUN mkdir -p src/fraud && touch src/fraud/__init__.py && pip install -e ".${EXTRAS}"

COPY src ./src
# Top-level streaming package (Kafka producer/consumer); importable since WORKDIR is on sys.path.
COPY streaming ./streaming

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --retries=5 --start-period=20s \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "fraud.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
