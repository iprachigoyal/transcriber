"""Worker entrypoint for the Talksy transcriber agent.

Run modes (provided by livekit-agents' CLI):

    python -m src.main dev      # connect to LiveKit with hot reload (development)
    python -m src.main start    # production
    python -m src.main connect --room <name>   # join a specific existing room

The worker registers under AGENT_NAME and is started ONLY when the host app
explicitly dispatches it into a room. We connect audio-only, never publish
media, caption each speaker, and disconnect cleanly when the room empties.
"""

from __future__ import annotations

import asyncio
import logging
import os

from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli

from .config import Config
from .pipeline import TranscriptionPipeline
from .stt.hf_whisper import HuggingFaceWhisperSpace

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("transcriber")

KEEP_WARM_INTERVAL_S = 240  # ping the Space every 4 minutes while a room is active


def _agent_name() -> str:
    return os.getenv("AGENT_NAME", "talksy-transcriber")


async def _keep_warm(stt: HuggingFaceWhisperSpace, stop: asyncio.Event) -> None:
    """Periodically ping the Space so a long-running room rarely hits a cold start."""
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=KEEP_WARM_INTERVAL_S)
        except asyncio.TimeoutError:
            await stt.health()


async def entrypoint(ctx: JobContext) -> None:
    cfg = Config.load()
    log.info("Dispatched into room=%s as agent=%s", ctx.room.name, cfg.agent_name)

    # Audio only: we never need video, and we never publish anything ourselves.
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    stt = HuggingFaceWhisperSpace(
        base_url=cfg.hf_whisper_url,
        token=cfg.hf_token,
        timeout_s=cfg.hf_timeout_s,
    )
    pipeline = TranscriptionPipeline(ctx.room, stt, cfg)
    pipeline.start()

    # Operability logging: see who is in the room and what they publish. This
    # makes it obvious whether the agent is co-located with the speakers and
    # whether their mic tracks are being seen.
    log.info(
        "connected; remote participants now: %s",
        [(p.identity, str(p.kind)) for p in ctx.room.remote_participants.values()],
    )

    def _on_participant_connected(p: rtc.RemoteParticipant) -> None:
        log.info("participant connected identity=%s kind=%s", p.identity, p.kind)

    def _on_track_published(
        pub: rtc.RemoteTrackPublication, p: rtc.RemoteParticipant
    ) -> None:
        log.info(
            "track published by=%s source=%s kind=%s",
            p.identity,
            pub.source,
            pub.kind,
        )

    ctx.room.on("participant_connected", _on_participant_connected)
    ctx.room.on("track_published", _on_track_published)

    stop = asyncio.Event()
    warm_task = asyncio.create_task(_keep_warm(stt, stop))

    async def _cleanup() -> None:
        stop.set()
        warm_task.cancel()
        try:
            await warm_task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
        await pipeline.aclose()
        await stt.aclose()

    ctx.add_shutdown_callback(_cleanup)

    # Disconnect cleanly once the last human leaves so we never linger.
    def _on_participant_disconnected(_p) -> None:
        if len(ctx.room.remote_participants) == 0:
            log.info("Room empty, shutting down job for room=%s", ctx.room.name)
            ctx.shutdown(reason="room empty")

    ctx.room.on("participant_disconnected", _on_participant_disconnected)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            # Explicit dispatch: the worker is only started when the host app
            # calls createDispatch(room, agent_name=AGENT_NAME). With agent_name
            # set, the worker does NOT auto-join every room.
            agent_name=_agent_name(),
        )
    )
