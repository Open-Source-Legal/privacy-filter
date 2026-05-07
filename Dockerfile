# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS builder

ARG MODEL_ID=openai/privacy-filter
ARG MODEL_REVISION=

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    HF_HOME=/opt/hf-cache

RUN pip install --no-cache-dir uv==0.4.20

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN uv sync --frozen --no-dev --extra hf --no-editable

# Bake model weights into the image so cold starts and air-gapped deployments
# do not depend on outbound network. The cache is later copied into the
# runtime stage and served via HF_HOME.
COPY --chmod=755 docker/bake-model.py /usr/local/bin/bake-model.py
RUN MODEL_ID="${MODEL_ID}" MODEL_REVISION="${MODEL_REVISION}" \
    /app/.venv/bin/python /usr/local/bin/bake-model.py

FROM python:3.12-slim AS runtime

RUN groupadd --system --gid 1001 app \
 && useradd  --system --uid 1001 --gid app --no-create-home app

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder --chown=app:app /opt/hf-cache /opt/hf-cache

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf-cache \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).status==200 else 1)"

CMD ["uvicorn", "privacy_filter.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
