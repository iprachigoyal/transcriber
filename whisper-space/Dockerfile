FROM python:3.11-slim

# ffmpeg is used by faster-whisper to decode various audio containers.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default model; override with a Space variable if you want tiny/small/base.
# base.en is the English-specific model: same speed as base on CPU, more accurate.
ENV WHISPER_MODEL=base.en
ENV WHISPER_LANGUAGE=en
ENV WHISPER_DEVICE=cpu
ENV WHISPER_COMPUTE_TYPE=int8

# Hugging Face Spaces (docker SDK) route traffic to this port.
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
