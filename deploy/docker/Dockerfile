# CPU image: API server + web viewer + CPU workers.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# No system packages needed: OpenCV comes from opencv-python-headless (no libGL/glib).
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN pip install ".[cloud,postgres,mongo]"

RUN useradd --create-home --uid 1000 phenoscope
USER phenoscope

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
ENTRYPOINT ["phenoscope"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
