FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CHAPTERVAULT_DATA=/data \
    CHAPTERVAULT_HOST=0.0.0.0

# LibreOffice converts Word files to PDF with their original layout; the fonts keep that layout close.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libreoffice-writer-nogui fonts-dejavu-core fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY run.py .
COPY vault ./vault

RUN useradd --system --create-home --home-dir /home/vault vault \
    && mkdir /data && chown vault /data
USER vault
VOLUME /data
EXPOSE 8000
CMD ["python", "run.py"]
