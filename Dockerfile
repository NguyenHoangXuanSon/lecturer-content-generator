#dockerhub
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y tesseract-ocr libtesseract-dev libleptonica-dev pkg-config poppler-utils && \
    rm -rf /var/lib/apt/lists/*

COPY requirements-base.txt .
RUN pip install --no-cache-dir -r requirements-base.txt

COPY requirements-extra.txt .
RUN pip install --no-cache-dir -r requirements-extra.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.app:app", "--reload", "--host", "0.0.0.0"]

