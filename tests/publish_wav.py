"""Stand-in for a human speaker: publish a WAV file as a microphone track.

Joins the room with a human-style identity and publishes the given 16 kHz mono
PCM16 WAV as a SOURCE_MICROPHONE track, paced in real time. The transcriber
agent only subscribes to microphone audio, so this is exactly what it captions.

After the clip we push ~1s of silence so the agent's VAD cleanly finalizes the
utterance, then hold the connection briefly so captions can come back.

Usage:
    .venv/bin/python tests/publish_wav.py --room demo --identity alice \
        --wav tests/fixtures/sample.wav
"""

from __future__ import annotations

import argparse
import asyncio
import os
import wave

from dotenv import load_dotenv
from livekit import api, rtc

load_dotenv()

FRAME_MS = 10  # 10 ms frames are a common, low-latency capture size


def _token(room: str, identity: str) -> str:
    return (
        api.AccessToken(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
        .with_identity(identity)
        .with_name(identity)
        .with_grants(api.VideoGrants(room_join=True, room=room, can_publish=True,
                                     can_subscribe=True))
        .to_jwt()
    )


def _read_wav(path: str) -> tuple[bytes, int]:
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2:
            raise SystemExit("WAV must be 16-bit PCM")
        if w.getnchannels() != 1:
            raise SystemExit("WAV must be mono")
        rate = w.getframerate()
        pcm = w.readframes(w.getnframes())
    return pcm, rate


async def _stream_pcm(source: rtc.AudioSource, pcm: bytes, rate: int) -> None:
    samples_per_frame = int(rate * FRAME_MS / 1000)
    frame_bytes = samples_per_frame * 2  # int16 mono
    for off in range(0, len(pcm) - frame_bytes + 1, frame_bytes):
        chunk = pcm[off:off + frame_bytes]
        frame = rtc.AudioFrame(
            data=chunk,
            sample_rate=rate,
            num_channels=1,
            samples_per_channel=samples_per_frame,
        )
        # capture_frame paces playback in real time via the source's queue.
        await source.capture_frame(frame)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", default="demo")
    ap.add_argument("--identity", default="alice")
    ap.add_argument("--wav", default="tests/fixtures/sample.wav")
    ap.add_argument("--repeat", type=int, default=1, help="times to play the clip")
    ap.add_argument("--hold", type=float, default=3.0,
                    help="seconds to stay connected after the audio")
    args = ap.parse_args()

    pcm, rate = _read_wav(args.wav)
    url = os.environ["LIVEKIT_URL"]

    room = rtc.Room()
    await room.connect(url, _token(args.room, args.identity))
    print(f"[pub] connected to '{args.room}' as '{args.identity}' "
          f"({rate} Hz, {len(pcm)//2/rate:.2f}s clip x{args.repeat})", flush=True)

    source = rtc.AudioSource(sample_rate=rate, num_channels=1)
    track = rtc.LocalAudioTrack.create_audio_track("mic", source)
    opts = rtc.TrackPublishOptions()
    opts.source = rtc.TrackSource.SOURCE_MICROPHONE
    await room.local_participant.publish_track(track, opts)
    print("[pub] published microphone track, streaming audio ...", flush=True)

    silence = b"\x00\x00" * int(rate * 1.0)  # 1s of trailing silence
    for i in range(args.repeat):
        await _stream_pcm(source, pcm, rate)
        await _stream_pcm(source, silence, rate)
        print(f"[pub] finished play {i + 1}/{args.repeat}", flush=True)

    await asyncio.sleep(args.hold)
    await room.disconnect()
    print("[pub] disconnected", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
