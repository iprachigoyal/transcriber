"""Dispatch the transcriber agent into a room - what the Talksy backend does.

When a room call starts, Talksy calls the equivalent of:

    AgentDispatchClient.createDispatch(room=<conversationId>, agent_name="talksy-transcriber")

This script does the same, then prints the room's participants so you can see
the agent (ParticipantKind.AGENT) join alongside the humans.

Usage:
    .venv/bin/python tests/dispatch_agent.py --room demo
"""

from __future__ import annotations

import argparse
import asyncio
import os

from dotenv import load_dotenv
from livekit import api

load_dotenv("/Users/prachigoyal/Projects/transcriber/.env")


def _http_url() -> str:
    return os.environ["LIVEKIT_URL"].replace("ws://", "http://").replace("wss://", "https://")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", default="demo")
    ap.add_argument("--agent-name", default=os.getenv("AGENT_NAME", "talksy-transcriber"))
    ap.add_argument("--list-after", type=float, default=2.0,
                    help="seconds to wait before listing participants")
    args = ap.parse_args()

    lk = api.LiveKitAPI(_http_url(), os.environ["LIVEKIT_API_KEY"],
                        os.environ["LIVEKIT_API_SECRET"])
    try:
        d = await lk.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(room=args.room, agent_name=args.agent_name)
        )
        print(f"[dispatch] created id={d.id} room={args.room} agent={args.agent_name}",
              flush=True)

        await asyncio.sleep(args.list_after)
        parts = await lk.room.list_participants(
            api.ListParticipantsRequest(room=args.room)
        )
        print(f"[dispatch] participants in '{args.room}':", flush=True)
        for p in parts.participants:
            # kind 4 == AGENT in the LiveKit proto enum.
            print(f"    identity={p.identity} name={p.name} kind={p.kind} "
                  f"tracks={[t.source for t in p.tracks]}", flush=True)
    finally:
        await lk.aclose()


if __name__ == "__main__":
    asyncio.run(main())
