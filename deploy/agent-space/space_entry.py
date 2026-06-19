"""Hugging Face Space entrypoint for the Talksy transcriber agent.

HF Docker Spaces expect an HTTP app on ``app_port`` (7860) and free Spaces
sleep when idle, so this wrapper does two jobs that a bare worker cannot:

1. Runs the LiveKit worker (``python -m src.main start``) as a child process,
   restarting it if it ever exits (Spaces have no systemd ``Restart=always``).
2. Serves a tiny ``/health`` endpoint so HF marks the Space healthy and an
   external uptime pinger can keep it awake.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time

import uvicorn
from fastapi import FastAPI

app = FastAPI(title="Talksy Transcriber Agent (Space wrapper)")
_proc: subprocess.Popen | None = None


def _spawn() -> subprocess.Popen:
    return subprocess.Popen(["python", "-m", "src.main", "start"])


def _watchdog() -> None:
    global _proc
    _proc = _spawn()
    while True:
        time.sleep(10)
        if _proc.poll() is not None:  # worker exited -> restart it
            _proc = _spawn()


# GET + HEAD: uptime monitors (e.g. UptimeRobot) probe with HEAD by default.
@app.api_route("/", methods=["GET", "HEAD"])
@app.api_route("/health", methods=["GET", "HEAD"])
def health() -> dict:
    alive = _proc is not None and _proc.poll() is None
    return {"status": "ok" if alive else "worker_down", "worker_alive": alive}


threading.Thread(target=_watchdog, daemon=True).start()
uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
