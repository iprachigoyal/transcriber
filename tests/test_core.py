"""Offline self-tests for the dependency-light core logic.

These do NOT need LiveKit or a live Whisper Space. They prove:
  * audio.py encodes valid WAV and resamples/downmixes correctly
  * vad.py segments utterances and drops too-short blips
  * stt/hf_whisper.py parses success responses and fails safe on errors

Run with:  .venv/bin/python -m pytest -q   (or)   .venv/bin/python tests/test_core.py
"""

from __future__ import annotations

import asyncio
import io
import os
import sys
import wave

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audio import encode_wav, resample_mono_int16, to_mono_int16  # noqa: E402
from src.stt.hf_whisper import HuggingFaceWhisperSpace  # noqa: E402
from src.vad import VadSegmenter  # noqa: E402


def test_encode_wav_roundtrip() -> None:
    pcm = (b"\x01\x00" * 16000)  # 1s of 16 kHz mono int16
    wav = encode_wav(pcm, sample_rate=16000, channels=1)
    with wave.open(io.BytesIO(wav), "rb") as w:
        assert w.getframerate() == 16000
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getnframes() == 16000
    print("ok: encode_wav roundtrip")


def test_resample_halves_samples() -> None:
    pcm = b"\x00\x01" * 1600  # 1600 mono samples @16k = 100ms
    out = resample_mono_int16(pcm, 16000, 8000)
    assert len(out) // 2 == 800, len(out) // 2
    same = resample_mono_int16(pcm, 16000, 16000)
    assert same == pcm
    print("ok: resample halves samples")


def test_to_mono_downmix() -> None:
    stereo = b"\x10\x00\x20\x00" * 100  # L=16, R=32 interleaved, 100 frames
    mono = to_mono_int16(stereo, channels=2)
    assert len(mono) // 2 == 100
    print("ok: stereo -> mono")


class _FakeVad:
    """Deterministic VAD: returns speech per a scripted boolean list."""

    def __init__(self, script: list[bool]) -> None:
        self._script = script
        self._i = 0

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        val = self._script[self._i] if self._i < len(self._script) else False
        self._i += 1
        return val


def _make_segmenter(script: list[bool]) -> VadSegmenter:
    seg = VadSegmenter(
        sample_rate=16000, frame_ms=20, silence_ms=600, max_ms=10000, min_ms=300
    )
    seg._vad = _FakeVad(script)  # type: ignore[assignment]
    return seg


def test_vad_emits_utterance() -> None:
    seg = _make_segmenter([True] * 20 + [False] * 30)  # 30 silent frames = 600ms
    frame = b"\x00\x00" * 320  # 20ms @16k mono
    emitted = None
    for _ in range(50):
        out = seg.push(frame)
        if out is not None:
            emitted = out
    assert emitted is not None, "expected an utterance"
    assert len(emitted) == 50 * len(frame)  # 20 voiced + 30 trailing-silence frames
    print("ok: vad emits utterance")


def test_vad_drops_short_blip() -> None:
    seg = _make_segmenter([True] * 5 + [False] * 30)  # only 100ms of speech (<300ms)
    frame = b"\x00\x00" * 320
    results = [seg.push(frame) for _ in range(35)]
    assert all(r is None for r in results), "short blip should be dropped"
    print("ok: vad drops short blip")


def test_hf_success() -> None:
    async def run() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/transcribe"
            return httpx.Response(200, json={"text": "hello there", "language": "en"})

        stt = HuggingFaceWhisperSpace("https://example.hf.space")
        stt._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        text, lang = await stt.transcribe(encode_wav(b"\x00\x00" * 1600))
        assert (text, lang) == ("hello there", "en"), (text, lang)
        await stt.aclose()

    asyncio.run(run())
    print("ok: hf success parses text/language")


def test_hf_failsafe() -> None:
    async def run() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="boom")

        stt = HuggingFaceWhisperSpace("https://example.hf.space", retries=1)
        stt._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        text, lang = await stt.transcribe(encode_wav(b"\x00\x00" * 1600))
        assert (text, lang) == ("", ""), "errors must fail safe to empty"
        await stt.aclose()

    asyncio.run(run())
    print("ok: hf fails safe on errors")


if __name__ == "__main__":
    test_encode_wav_roundtrip()
    test_resample_halves_samples()
    test_to_mono_downmix()
    test_vad_emits_utterance()
    test_vad_drops_short_blip()
    test_hf_success()
    test_hf_failsafe()
    print("\nALL CORE TESTS PASSED")
