FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

COPY requirements-api.txt .
RUN pip install --no-cache-dir --timeout=120 -r requirements-api.txt

COPY api/ ./api/
COPY src/ ./src/
COPY models/ ./models/
COPY .env.example .env

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]