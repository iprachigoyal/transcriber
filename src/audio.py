"""Audio helpers: downmix to mono, resample, and encode PCM16 WAV in memory.

LiveKit hands us raw PCM frames. Whisper wants a 16 kHz mono PCM16 WAV. These
small, dependency-light helpers bridge the two. We let LiveKit's AudioStream do
the heavy resampling when possible (see pipeline.py), but keep a pure-Python
resampler here as a fallback and for clarity.
"""

from __future__ import annotations

import io
import wave

import numpy as np


def to_mono_int16(pcm16: bytes, channels: int) -> bytes:
    """Average interleaved channels down to a single mono channel."""
    if channels <= 1:
        return pcm16
    samples = np.frombuffer(pcm16, dtype=np.int16)
    # Drop any trailing partial frame so reshape is exact.
    usable = (samples.size // channels) * channels
    samples = samples[:usable].reshape(-1, channels)
    mono = samples.mean(axis=1)
    return mono.astype(np.int16).tobytes()


def resample_mono_int16(pcm16: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Linear-interpolation resample of mono PCM16. Good enough for speech."""
    if src_rate == dst_rate or not pcm16:
        return pcm16
    samples = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return b""
    duration = samples.size / src_rate
    dst_n = max(1, int(round(duration * dst_rate)))
    x_old = np.linspace(0.0, duration, num=samples.size, endpoint=False)
    x_new = np.linspace(0.0, duration, num=dst_n, endpoint=False)
    resampled = np.interp(x_new, x_old, samples)
    return np.clip(resampled, -32768, 32767).astype(np.int16).tobytes()


def encode_wav(pcm16: bytes, sample_rate: int = 16000, channels: int = 1) -> bytes:
    """Wrap raw PCM16 little-endian samples in a WAV container, in memory."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)  # 16-bit
        w.setframerate(sample_rate)
        w.writeframes(pcm16)
    return buf.getvalue()
