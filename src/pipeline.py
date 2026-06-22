"""The per-speaker transcription pipeline.

For every remote participant's microphone track we run an isolated task that:

    AudioStream (16 kHz mono PCM) -> slice into 20 ms frames -> VAD segmentation
    -> utterance queue -> Whisper -> publish_transcription attributed to the speaker

Key design choices and why:

* One task per speaker, fully isolated. If one speaker's STT errors or their
  track misbehaves, everyone else keeps getting captions.
* A small bounded queue per speaker decouples "listening" from "transcribing".
  If Whisper falls behind (cold/slow), we drop the OLDEST queued utterance
  rather than grow memory without bound or block the audio loop.
* Captions are attributed to the SPEAKER's identity + mic track sid (never the
  agent's), which is what the Talksy browser UI renders against.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid

from livekit import rtc

from .audio import encode_wav
from .config import Config
from .stt.base import SttBackend
from .vad import VadSegmenter

log = logging.getLogger("transcriber.pipeline")

FRAME_MS = 20
# Buffer enough short (~4s) utterances to absorb bursts and brief STT slowness
# without dropping speech. Smaller segments clear faster, so a backlog is rare;
# this headroom means we rarely have to discard anything.
MAX_QUEUED_UTTERANCES = 12

# Reliable data-channel topic for captions. We publish each caption here IN ADDITION to the
# native publish_transcription, because some browser livekit-client builds don't decode the
# transcription wire format this agent's server SDK emits and would silently show nothing.
# The Talksy web client renders captions from this topic. Must match CAPTION_TOPIC on the client.
CAPTION_TOPIC = "talksy-captions"


class TranscriptionPipeline:
    def __init__(self, room: rtc.Room, stt: SttBackend, cfg: Config) -> None:
        self._room = room
        self._stt = stt
        self._cfg = cfg
        # keyed by track sid -> the speaker task
        self._tasks: dict[str, asyncio.Task] = {}

    def start(self) -> None:
        self._room.on("track_subscribed", self._on_track_subscribed)
        self._room.on("track_unsubscribed", self._on_track_unsubscribed)
        # Pick up any tracks that were already subscribed before we attached.
        for participant in self._room.remote_participants.values():
            for pub in participant.track_publications.values():
                if pub.track is not None:
                    self._maybe_start(pub.track, pub, participant)

    async def aclose(self) -> None:
        for task in list(self._tasks.values()):
            task.cancel()
        for task in list(self._tasks.values()):
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._tasks.clear()

    # --- event handlers -----------------------------------------------------

    def _on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        self._maybe_start(track, publication, participant)

    def _on_track_unsubscribed(
        self,
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        task = self._tasks.pop(publication.sid, None)
        if task is not None:
            task.cancel()

    def _maybe_start(
        self,
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        if publication.source != rtc.TrackSource.SOURCE_MICROPHONE:
            return
        sid = publication.sid
        if sid in self._tasks:
            return
        task = asyncio.create_task(
            self._run_speaker(track, sid, participant.identity),
            name=f"speaker:{participant.identity}",
        )
        self._tasks[sid] = task

    # --- per-speaker work ---------------------------------------------------

    async def _run_speaker(self, track: rtc.Track, track_sid: str, identity: str) -> None:
        log.info("Start captioning identity=%s track=%s", identity, track_sid)
        segmenter = VadSegmenter(
            sample_rate=self._cfg.sample_rate,
            frame_ms=FRAME_MS,
            silence_ms=self._cfg.vad_silence_ms,
            max_ms=self._cfg.segment_max_ms,
            min_ms=self._cfg.segment_min_ms,
            aggressiveness=self._cfg.vad_aggressiveness,
        )
        frame_bytes = segmenter.frame_bytes
        queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=MAX_QUEUED_UTTERANCES)

        consumer = asyncio.create_task(self._consume(queue, identity, track_sid))
        stream = rtc.AudioStream.from_track(
            track=track, sample_rate=self._cfg.sample_rate, num_channels=1
        )
        leftover = b""
        try:
            async for event in stream:
                # AudioStream already resampled to 16 kHz mono PCM16 for us.
                leftover += bytes(event.frame.data)
                while len(leftover) >= frame_bytes:
                    frame = leftover[:frame_bytes]
                    leftover = leftover[frame_bytes:]
                    utterance = segmenter.push(frame)
                    if utterance:
                        self._enqueue(queue, utterance)
            # Stream ended: flush a trailing utterance if any.
            tail = segmenter.flush()
            if tail:
                self._enqueue(queue, tail)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - isolate this speaker
            log.warning("Speaker loop error identity=%s: %s", identity, exc)
        finally:
            consumer.cancel()
            try:
                await consumer
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            await stream.aclose()
            self._tasks.pop(track_sid, None)
            log.info("Stop captioning identity=%s track=%s", identity, track_sid)

    def _enqueue(self, queue: "asyncio.Queue[bytes]", utterance: bytes) -> None:
        if queue.full():
            # Drop the oldest queued utterance so we never grow unbounded and
            # always prefer the most recent speech.
            try:
                queue.get_nowait()
                log.warning("STT backlog: dropped an older utterance (captions may be missing)")
            except asyncio.QueueEmpty:
                pass
        try:
            queue.put_nowait(utterance)
        except asyncio.QueueFull:
            pass

    async def _consume(
        self, queue: "asyncio.Queue[bytes]", identity: str, track_sid: str
    ) -> None:
        while True:
            utterance = await queue.get()
            try:
                wav = encode_wav(utterance, sample_rate=self._cfg.sample_rate, channels=1)
                text, language = await self._stt.transcribe(wav)
                if text:
                    await self._publish(identity, track_sid, text, language)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.warning("Transcribe/publish failed identity=%s: %s", identity, exc)
            finally:
                queue.task_done()

    async def _publish(
        self, identity: str, track_sid: str, text: str, language: str
    ) -> None:
        segment = rtc.TranscriptionSegment(
            id=uuid.uuid4().hex,
            text=text,
            start_time=0,
            end_time=0,
            final=True,
            language=language or "",
        )
        # 1) Native transcription (for LiveKit clients that decode it).
        try:
            await self._room.local_participant.publish_transcription(
                rtc.Transcription(
                    participant_identity=identity,
                    track_sid=track_sid,
                    segments=[segment],
                )
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("publish_transcription failed: %s", exc)

        # 2) Reliable data-channel caption (what the Talksy web client actually renders).
        try:
            await self._room.local_participant.publish_data(
                json.dumps(
                    {
                        "id": segment.id,
                        "identity": identity,
                        "text": text,
                        "language": language or "",
                        "final": True,
                    }
                ),
                reliable=True,
                topic=CAPTION_TOPIC,
            )
            log.info("caption identity=%s [%s] %s", identity, language, text)
        except Exception as exc:  # noqa: BLE001
            log.warning("publish_data caption failed: %s", exc)
