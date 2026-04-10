# Python 3.10 aligns well with Paddle wheels; slim keeps the image smaller.
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# OpenCV / GUI-ish libs, OpenMP (Paddle), GL for cv2, and Tesseract binary
# (pytesseract is only a wrapper — without tesseract-ocr, Tesseract OCR fails at runtime).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    libgomp1 \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

EXPOSE 8000

# Run from /app so `src.api:app` resolves (project root = WORKDIR).
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
