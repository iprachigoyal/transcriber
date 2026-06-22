"""Environment configuration loading and validation.

Everything the service needs is read from environment variables (or a local
``.env`` file). We validate up front and fail with a clear message so a
misconfigured deploy never limps along silently.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load a local .env if present. In production the env is set by the host.
load_dotenv()


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or malformed."""


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ConfigError(f"Missing required environment variable: {name}")
    return value


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


@dataclass(frozen=True)
class Config:
    livekit_url: str
    livekit_api_key: str
    livekit_api_secret: str
    agent_name: str
    hf_whisper_url: str
    hf_token: str | None
    stt_language: str | None
    sample_rate: int
    vad_silence_ms: int
    segment_max_ms: int
    segment_min_ms: int
    vad_aggressiveness: int
    hf_timeout_s: float

    @staticmethod
    def load() -> "Config":
        sample_rate = _int("SAMPLE_RATE", 16000)
        # webrtcvad only supports 8000/16000/32000/48000 Hz.
        if sample_rate not in (8000, 16000, 32000, 48000):
            raise ConfigError(
                f"SAMPLE_RATE must be one of 8000/16000/32000/48000, got {sample_rate}"
            )

        aggressiveness = _int("VAD_AGGRESSIVENESS", 2)
        if aggressiveness not in (0, 1, 2, 3):
            raise ConfigError("VAD_AGGRESSIVENESS must be between 0 and 3")

        return Config(
            livekit_url=_require("LIVEKIT_URL"),
            livekit_api_key=_require("LIVEKIT_API_KEY"),
            livekit_api_secret=_require("LIVEKIT_API_SECRET"),
            agent_name=os.getenv("AGENT_NAME", "talksy-transcriber"),
            hf_whisper_url=_require("HF_WHISPER_URL"),
            hf_token=os.getenv("HF_TOKEN") or None,
            stt_language=(os.getenv("STT_LANGUAGE", "en") or "").strip() or None,
            sample_rate=sample_rate,
            vad_silence_ms=_int("VAD_SILENCE_MS", 500),
            segment_max_ms=_int("SEGMENT_MAX_MS", 4000),
            segment_min_ms=_int("SEGMENT_MIN_MS", 300),
            vad_aggressiveness=aggressiveness,
            hf_timeout_s=_float("HF_TIMEOUT_S", 15.0),
        )
