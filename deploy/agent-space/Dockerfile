FROM python:3.11-slim

# build-essential + python3-dev: webrtcvad compiles a C extension at install.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
COPY requirements-space.txt .
RUN pip install --no-cache-dir -r requirements.txt -r requirements-space.txt

COPY src ./src
COPY space_entry.py .

# HF Spaces route traffic to 7860; the health server listens there.
ENV PORT=7860
EXPOSE 7860
CMD ["python", "space_entry.py"]
