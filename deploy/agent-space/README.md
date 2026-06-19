---
title: Talksy Transcriber Agent
emoji: "🎤"
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Talksy Transcriber Agent (Hugging Face Space)

Hosts the long-running LiveKit transcriber **worker** from
[`transcriber`](https://github.com/) on a free HF Docker Space.

`space_entry.py` runs `python -m src.main start` (the worker) under a watchdog
and exposes a `/health` endpoint on port 7860 so the Space stays healthy and an
external uptime pinger can keep it awake.

## Configure

Set these in **Settings -> Variables and secrets** (never commit them):

- `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`
- `AGENT_NAME` (default `talksy-transcriber`)
- `HF_WHISPER_URL` (e.g. `https://iprachigoyal-talksy-whisper.hf.space`)
- `HF_TOKEN` (only if the Whisper Space is private)

## Keep awake

Point a free uptime monitor (UptimeRobot / cron-job.org) at
`https://<owner>-talksy-transcriber-agent.hf.space/health` every ~5 minutes so
the free Space does not go idle and the worker stays registered with LiveKit.
