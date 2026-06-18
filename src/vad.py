"""Voice-activity-based utterance segmentation using webrtcvad.

You feed this fixed-size PCM16 frames (10/20/30 ms at a supported rate). It
batches consecutive speech frames into an "utterance" and emits the utterance
when it sees enough trailing silence (someone stopped talking) or when the
utterance hits a maximum length (a long monologue should still produce timely
captions).

Why fixed frame sizes? webrtcvad only accepts 10, 20, or 30 ms frames at
8000/16000/32000/48000 Hz. We use 20 ms frames, which at 16 kHz is exactly
320 samples = 640 bytes.
"""

from __future__ import annotations

import webrtcvad


class VadSegmenter:
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 20,
        silence_ms: int = 600,
        max_ms: int = 10000,
        min_ms: int = 300,
        aggressiveness: int = 2,
    ) -> None:
        if frame_ms not in (10, 20, 30):
            raise ValueError("frame_ms must be 10, 20, or 30")
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_bytes = int(sample_rate * (frame_ms / 1000.0)) * 2  # 16-bit mono
        self._silence_frames_needed = max(1, silence_ms // frame_ms)
        self._max_frames = max(1, max_ms // frame_ms)
        self._min_frames = max(1, min_ms // frame_ms)

        self._vad = webrtcvad.Vad(aggressiveness)
        self._reset()

    def _reset(self) -> None:
        self._triggered = False
        self._frames: list[bytes] = []
        self._trailing_silence = 0

    def push(self, frame: bytes) -> bytes | None:
        """Feed one fixed-size frame. Returns a completed utterance or None.

        The frame MUST be exactly ``frame_bytes`` long (the caller slices the
        incoming audio into frames; see pipeline.py).
        """
        if len(frame) != self.frame_bytes:
            # Ignore malformed frames rather than crash the stream.
            return None

        is_speech = self._vad.is_speech(frame, self.sample_rate)

        if not self._triggered:
            if is_speech:
                self._triggered = True
                self._frames = [frame]
                self._trailing_silence = 0
            return None

        # We are inside an utterance.
        self._frames.append(frame)
        if is_speech:
            self._trailing_silence = 0
        else:
            self._trailing_silence += 1

        if self._trailing_silence >= self._silence_frames_needed:
            return self._emit()
        if len(self._frames) >= self._max_frames:
            return self._emit()
        return None

    def flush(self) -> bytes | None:
        """Emit any in-progress utterance (e.g. when a speaker disconnects)."""
        if self._triggered and self._frames:
            return self._emit()
        return None

    def _emit(self) -> bytes | None:
        voiced = len(self._frames) - self._trailing_silence
        pcm = b"".join(self._frames)
        self._reset()
        if voiced < self._min_frames:
            return None  # too short, almost certainly noise
        return pcm
