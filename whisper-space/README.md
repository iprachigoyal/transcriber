---
title: Talksy Whisper
emoji: "🎙️"
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Talksy Whisper Space

A tiny FastAPI service that transcribes audio with
[`faster-whisper`](https://github.com/SYSTRAN/faster-whisper). It powers live
captions for the Talksy transcriber agent.

## API

### `POST /transcribe`

`multipart/form-data` with a single field `file` containing a 16 kHz mono PCM16
WAV. If the Space is private, send `Authorization: Bearer <SPACE_TOKEN>`.

```bash
curl -s -X POST "$HF_WHISPER_URL/transcribe" \
  -F "file=@sample.wav;type=audio/wav"
# -> {"text":"hello there","language":"en"}
```

### `GET /health`

Returns `200` with model status. Used by the agent for keep-warm pings.

## Configuration (Space variables / env)

| Variable               | Default   | Notes                                            |
| ---------------------- | --------- | ------------------------------------------------ |
| `WHISPER_MODEL`        | `base.en` | English-specific; `base` for multilingual, `small.en` for quality. |
| `WHISPER_LANGUAGE`     | `en`      | Pinned language; empty = auto-detect (less accurate on short clips). |
| `WHISPER_DEVICE`       | `cpu`     | Free Spaces are CPU-only.                        |
| `WHISPER_COMPUTE_TYPE` | `int8`    | Fast + low memory on CPU.                        |
| `WHISPER_BEAM_SIZE`    | `1`       | Greedy decode; raise for slightly better text.   |
| `SPACE_TOKEN`          | (unset)   | If set, requests must carry the bearer token.    |

## Run locally

```bash
cd whisper-space
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

## Deploy to Hugging Face

1. Create a new Space, **SDK = Docker**.
2. Push this folder's contents (including `Dockerfile` and this `README.md`
   with the YAML header) to the Space repo.
3. First request after idle is a cold start (30-60s); the agent handles that.
