# API image. Runs in artifact mode unless a dataset and checkpoint are mounted in.
FROM python:3.12-slim

WORKDIR /app

# CPU-only torch: the CUDA wheels are several GB and nothing here needs a GPU.
RUN pip install --no-cache-dir torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY fraudlens ./fraudlens
COPY api ./api
COPY artifacts ./artifacts
RUN pip install --no-cache-dir --no-deps -e .

ENV FRAUDLENS_ARTIFACTS_DIR=/app/artifacts \
    FRAUDLENS_CACHE_DIR=/app/.cache \
    PYTHONUNBUFFERED=1

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/api/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
