FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt

COPY . .
RUN pip install --no-cache-dir -e .

CMD gunicorn extracto.web.app:app --bind 0.0.0.0:${PORT:-8080} --workers 1 --timeout 120
