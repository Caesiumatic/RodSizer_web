FROM python:3.10-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# System dependencies: OpenCV runtime libs + Tesseract OCR (used to read the
# burned-in scale bar / "Pixel size" footer of camera exports that have no
# embedded calibration metadata).
USER root
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*
USER user

COPY --chown=user backend/requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . /app

ENV PYTHONPATH="/app/backend"

EXPOSE 7860

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
