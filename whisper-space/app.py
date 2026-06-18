"""Hugging Face Whisper Space - a tiny FastAPI speech-to-text service.

Contract (the transcriber agent depends on this exact shape):

    POST /transcribe   multipart/form-data, field "file" = 16 kHz mono PCM16 WAV
        -> 200 {"text": "hello there", "language": "en"}

    GET  /health       -> 200 {"status": "ok", ...}   (keep-warm / readiness)

The model is loaded ONCE at startup and kept in memory (loading per request
would make every caption painfully slow). On a free CPU Space, "base" is a good
speed/quality balance; set WHISPER_MODEL=tiny if latency is too high.
"""

from __future__ import annotations

import io
import logging
import os

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("whisper-space")

MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")  # fast + small on CPU
BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "1"))
# Optional shared secret. If set, callers must send Authorization: Bearer <token>.
SPACE_TOKEN = os.getenv("SPACE_TOKEN") or None

app = FastAPI(title="Talksy Whisper Space")
_model: WhisperModel | None = None


@app.on_event("startup")
def _load_model() -> None:
    global _model
    log.info("Loading faster-whisper model=%s device=%s compute=%s", MODEL_SIZE, DEVICE, COMPUTE_TYPE)
    _model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    log.info("Model loaded.")


def _check_auth(authorization: str | None = Header(default=None)) -> None:
    if SPACE_TOKEN is None:
        return  # public Space
    expected = f"Bearer {SPACE_TOKEN}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="invalid or missing token")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_SIZE, "loaded": _model is not None}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    _: None = Depends(_check_auth),
) -> dict:
    if _model is None:
        raise HTTPException(status_code=503, detail="model not loaded yet")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="empty audio")

    # faster-whisper accepts a file-like object; no temp file needed.
    segments, info = _model.transcribe(io.BytesIO(audio_bytes), beam_size=BEAM_SIZE)
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return {"text": text, "language": info.language}
