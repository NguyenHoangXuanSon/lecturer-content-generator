FROM python:3.12-bookworm

RUN apt-get update && apt-get install --no-install-recommends -y \
        build-essential \
        curl \
        ca-certificates \
        libglib2.0-0 \
        tesseract-ocr \
        libtesseract-dev \
        libjpeg-dev \
        zlib1g-dev \
        libfreetype6-dev \
        liblcms2-dev \
        libwebp-dev \
        libharfbuzz-dev \
        libfribidi-dev \
        libxcb1 \
        libxkbcommon0 \
        ghostscript \
        poppler-utils\
        swig\
        libgl1-mesa-glx \
        libxrender1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ADD https://astral.sh/uv/install.sh /install.sh
RUN chmod +x /install.sh && /install.sh && rm /install.sh

ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

COPY ./requirements.txt .
RUN uv pip install -r requirements.txt --system

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.app:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]
