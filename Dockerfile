# CLIP + FAISS cover recognizer. No system libs beyond Python are needed (Pillow, faiss-cpu,
# and torch all ship as wheels).
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# The CLIP model is downloaded from the Hugging Face hub on first use; point its cache at a
# writable dir. Also make the uploads dir writable (works as root or a non-root host user).
ENV HF_HOME=/tmp/hf
RUN mkdir -p static/uploads /tmp/hf && chmod -R 777 static/uploads /tmp/hf

EXPOSE 7860

# One worker keeps the (large) CLIP model loaded once; long timeout covers the first-request
# model download.
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "300", "app:app"]
