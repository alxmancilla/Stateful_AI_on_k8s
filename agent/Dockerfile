# syntax=docker/dockerfile:1
# Multi-stage build. Wheels are built in the builder stage and installed into a
# virtualenv that is copied into a slim runtime image (final image < 500MB).
#
# Build from the repository root so create_indexes.py can be included:
#   docker build -t stateful-agent:latest -f agent/Dockerfile .

FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY agent/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt


FROM python:3.11-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN useradd --create-home --uid 1000 agent
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY agent/ /app/
COPY scripts/create_indexes.py /app/create_indexes.py

USER agent
EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=3s --retries=5 \
    CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8080/health'); " || exit 1

ENTRYPOINT ["python", "main.py"]
CMD ["serve"]
