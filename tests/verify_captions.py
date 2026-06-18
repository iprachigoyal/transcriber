"""Stand-in for the Talksy browser: join a room and print captions.

The real Talksy UI listens for ``RoomEvent.TranscriptionReceived`` and renders
each segment under the participant it is attributed to. This script does exactly
that, so we can prove end-to-end that the agent publishes captions attributed to
the SPEAKER's identity (not the agent's).

Usage:
    .venv/bin/python tests/verify_captions.py --room demo --duration 60
"""

from __future__ import annotations

import argparse
import asyncio
import os

from dotenv import load_dotenv
from livekit import api, rtc

load_dotenv()


def _token(room: str, identity: str) -> str:
    return (
        api.AccessToken(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
        .with_identity(identity)
        .with_name(identity)
        .with_grants(api.VideoGrants(room_join=True, room=room, can_subscribe=True,
                                     can_publish=False))
        .to_jwt()
    )


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", default="demo")
    ap.add_argument("--identity", default="viewer")
    ap.add_argument("--duration", type=float, default=120.0,
                    help="seconds to listen before exiting")
    args = ap.parse_args()

    url = os.environ["LIVEKIT_URL"]
    room = rtc.Room()

    @room.on("transcription_received")
    def _on_tx(segments, participant, publication):  # noqa: ANN001
        who = participant.identity if participant else "<unknown>"
        for s in segments:
            state = "FINAL" if s.final else "interim"
            print(f"[caption] speaker={who} lang={s.language or '?'} "
                  f"({state}) track={getattr(publication, 'sid', '?')}: {s.text}",
                  flush=True)

    @room.on("participant_connected")
    def _on_join(p):  # noqa: ANN001
        print(f"[room] participant joined: identity={p.identity} kind={p.kind}", flush=True)

    await room.connect(url, _token(args.room, args.identity))
    print(f"[room] connected to '{args.room}' as '{args.identity}'. "
          f"Existing participants: "
          f"{[ (p.identity, str(p.kind)) for p in room.remote_participants.values() ]}",
          flush=True)
    print(f"[room] listening for captions for {args.duration:.0f}s ...", flush=True)

    try:
        await asyncio.sleep(args.duration)
    finally:
        await room.disconnect()
        print("[room] disconnected", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
