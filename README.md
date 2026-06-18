# Talksy Transcriber

A standalone, server-side **LiveKit transcription agent**. It joins a LiveKit
room when dispatched, listens to participants' microphone audio, transcribes
speech with a self-hosted **Hugging Face Whisper Space**, and publishes live
captions back into the room attributed to the real speaker.

It does exactly one job: **audio in -> captions published to the room**. No
translation, no storage, no database, no calls back into the host app.

```
Browser mic --> LiveKit SFU --> [this agent] --VAD--> WAV --> Whisper Space
                                     |
                                     +--> publish_transcription (as the speaker)
                                              |
                                          LiveKit SFU --> Browser shows caption
```

## Why these design choices (the integration contract)

These are the seams the Talksy app depends on; breaking them breaks captions:

1. **Joins as a LiveKit Agent** (`ParticipantKind.AGENT`) so Talksy hides it from
   the call grid. We use the `livekit-agents` worker model.
2. **Registered under a fixed agent name** (`AGENT_NAME`, default
   `talksy-transcriber`). Talksy dispatches it explicitly by name, so it only
   ever joins rooms it was told to.
3. **Never publishes audio/video** - listener + caption publisher only.
4. **Publishes captions via `publish_transcription`** attributed to the
   **speaker's identity + mic track sid**. We deliberately do NOT use
   `AgentSession`, because its newer text-stream transcription does not fire the
   browser `RoomEvent.TranscriptionReceived` that Talksy listens for.
5. **Disconnects cleanly** when the room empties.
6. **Best-effort / crash-safe** - a cold or broken Whisper Space just means "no
   caption for this utterance"; the call continues.

## Project layout

```
transcriber/
  requirements.txt
  .env.example
  README.md
  src/
    main.py          # worker entrypoint: WorkerOptions(agent_name=...), entrypoint(ctx)
    pipeline.py      # per-participant subscribe -> VAD -> WAV -> STT -> publish
    vad.py           # webrtcvad utterance segmentation
    audio.py         # resample + WAV(PCM16) encode helpers
    config.py        # env loading/validation
    stt/
      base.py        # SttBackend Protocol (the swappable seam)
      hf_whisper.py  # HuggingFaceWhisperSpace implementation
  whisper-space/     # the Hugging Face Whisper Space (deploy separately)
    app.py
    Dockerfile
    requirements.txt
    README.md
```

## Setup

> **Python 3.11+ is required.** `livekit-agents` uses syntax/typing that needs
> Python 3.10 or newer; 3.11+ is recommended. Check with `python3 --version`.
> On macOS you can install it with `brew install python@3.11` (then use
> `python3.11` below).

```bash
cd ~/Projects/transcriber
python3.11 -m venv .venv          # use a 3.11+ interpreter
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in the values
```

### Run the offline self-tests (no LiveKit / Space needed)

```bash
python tests/test_core.py
```

### Configuration

See `.env.example`. The values you must set:

| Variable             | What it is                                              |
| -------------------- | ------------------------------------------------------- |
| `LIVEKIT_URL`        | LiveKit server URL (from the Talksy operator).          |
| `LIVEKIT_API_KEY`    | LiveKit API key (from the operator).                    |
| `LIVEKIT_API_SECRET` | LiveKit API secret (from the operator).                 |
| `AGENT_NAME`         | Dispatch name; must match Talksy (`talksy-transcriber`).|
| `HF_WHISPER_URL`     | Base URL of your deployed Whisper Space.                |
| `HF_TOKEN`           | Only if your Space is private.                          |

Tuning knobs (`VAD_SILENCE_MS`, `SEGMENT_MAX_MS`, `SEGMENT_MIN_MS`,
`VAD_AGGRESSIVENESS`, `HF_TIMEOUT_S`) have sane defaults.

## Run

First deploy/run the Whisper Space (see `whisper-space/README.md`) and set
`HF_WHISPER_URL`. Then:

```bash
# development (connects to LiveKit, hot reload):
python -m src.main dev

# production:
python -m src.main start

# join a specific existing room manually (handy for testing):
python -m src.main connect --room <conversation-id>
```

In normal operation you do **not** pick rooms - the Talksy backend dispatches
the agent by name when a room call starts.

## How it works internally

1. Connect to the room audio-only (`AutoSubscribe.AUDIO_ONLY`).
2. For each remote participant's **microphone** track, open an
   `AudioStream(sample_rate=16000, num_channels=1)` - LiveKit resamples to
   Whisper's expected 16 kHz mono for us.
3. Slice the PCM into 20 ms frames and run **webrtcvad** to cut utterances
   (~600 ms trailing silence, or a 10 s max; sub-300 ms blips dropped).
4. Encode each utterance as an in-memory 16 kHz mono PCM16 WAV.
5. POST it to the Whisper Space; get back `{text, language}`.
6. `publish_transcription` attributed to that speaker's identity + mic track sid.

Each speaker runs in an isolated task with a small bounded queue: if Whisper
falls behind, the oldest queued utterance is dropped (bounded memory, freshest
captions), and one speaker's failure never affects the others.

## What Talksy needs back from you

- The **agent name** you registered (default `talksy-transcriber`).
- The **Whisper Space base URL** and whether it needs a token.
- Confirmation that the agent joins as **Agent kind** and publishes
  transcriptions attributed to the **speaker's identity + mic track sid**.
