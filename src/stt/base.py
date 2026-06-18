"""The pluggable speech-to-text seam.

The rest of the pipeline only knows about this interface, so the backend can be
swapped later (a local Whisper, a different cloud STT, etc.) without touching
the audio or LiveKit code.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class SttBackend(Protocol):
    async def transcribe(self, wav_bytes: bytes) -> tuple[str, str]:
        """Transcribe a 16 kHz mono PCM16 WAV.

        Returns ``(text, language)``. On any failure it must return
        ``("", "")`` rather than raising, so a flaky backend can never crash
        the audio pipeline.
        """
        ...
