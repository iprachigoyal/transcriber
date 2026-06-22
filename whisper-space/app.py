"""Hugging Face Whisper Space - a tiny FastAPI speech-to-text service.

Contract (the transcriber agent depends on this exact shape):

    POST /transcribe   multipart/form-data, field "file" = 16 kHz mono PCM16 WAV
        -> 200 {"text": "hello there", "language": "en"}

    GET  /health       -> 200 {"status": "ok", ...}   (keep-warm / readiness)

The model is loaded ONCE at startup and kept in memory (loading per request
would make every caption painfully slow). 
"""

from __future__ import annotations

import io
import logging
import os

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("whisper-space")

# Default to the English-specific model: same speed as `base` on CPU but more
# accurate for English audio. Override with WHISPER_MODEL for other use cases.
MODEL_SIZE = os.getenv("WHISPER_MODEL", "base.en")
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")  # fast + small on CPU
BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "1"))
# Pin a language so Whisper never guesses (per-clip auto-detection on short
# utterances frequently misfires and produces garbage). Empty string = auto.
DEFAULT_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en") or None
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
    language: str | None = Form(default=None),
    _: None = Depends(_check_auth),
) -> dict:
    if _model is None:
        raise HTTPException(status_code=503, detail="model not loaded yet")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="empty audio")

    # Pin language to the caller's hint, falling back to the Space default.
    # A pinned language stops Whisper guessing the wrong language on short clips.
    lang = (language or "").strip() or DEFAULT_LANGUAGE

    # faster-whisper accepts a file-like object; no temp file needed.
    # The guards below suppress the classic Whisper hallucinations on short or
    # near-silent clips (e.g. phantom "Thank you for watching" phrases):
    #   * vad_filter trims non-speech before decoding
    #   * condition_on_previous_text=False stops runaway repeats across clips
    #   * the no-speech / log-prob thresholds + temperature fallback drop
    #     low-confidence, likely-hallucinated output
    segments, info = _model.transcribe(
        io.BytesIO(audio_bytes),
        beam_size=BEAM_SIZE,
        language=lang,
        vad_filter=True,
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        log_prob_threshold=-1.0,
        temperature=[0.0, 0.2, 0.4],
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return {"text": text, "language": info.language}
