"""Speech-to-text backend backed by a self-hosted Hugging Face Whisper Space.

Contract with the Space (see whisper-space/app.py):

    POST {base_url}/transcribe   (multipart/form-data, field "file" = WAV)
    -> 200 {"text": "...", "language": "en"}

    GET  {base_url}/health       -> 200 (used for keep-warm)

This client is deliberately *fail-safe*: every error path returns ("", "") so a
cold, slow, or broken Space simply means "no caption for this utterance" and the
call keeps going.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

log = logging.getLogger("transcriber.stt")


class HuggingFaceWhisperSpace:
    def __init__(
        self,
        base_url: str,
        token: str | None = None,
        timeout_s: float = 15.0,
        cold_start_timeout_s: float = 60.0,
        retries: int = 1,
        language: str | None = None,
    ) -> None:
        base = base_url.rstrip("/")
        self._transcribe_url = f"{base}/transcribe"
        self._health_url = f"{base}/health"
        self._headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._timeout_s = timeout_s
        self._cold_start_timeout_s = cold_start_timeout_s
        self._retries = retries
        # Pinned language hint sent with every request so the Space never has to
        # guess the language on a short clip (a common source of wrong text).
        self._language = (language or "").strip() or None
        # One shared client; HTTP/2 keep-alive avoids reconnect overhead.
        self._client = httpx.AsyncClient(headers=self._headers)

    async def transcribe(self, wav_bytes: bytes) -> tuple[str, str]:
        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
        data = {"language": self._language} if self._language else None
        attempt = 0
        while attempt <= self._retries:
            # First try is short (warm Space); the retry allows for a cold start.
            timeout = self._timeout_s if attempt == 0 else self._cold_start_timeout_s
            try:
                resp = await self._client.post(
                    self._transcribe_url, files=files, data=data, timeout=timeout
                )
                resp.raise_for_status()
                data = resp.json()
                text = (data.get("text") or "").strip()
                language = data.get("language") or ""
                return text, language
            except Exception as exc:  # noqa: BLE001 - we intentionally swallow all
                attempt += 1
                if attempt > self._retries:
                    log.warning("Whisper Space failed, dropping segment: %s", exc)
                    return "", ""
                log.info(
                    "Whisper Space call failed (attempt %d), retrying "
                    "(maybe cold start): %s",
                    attempt,
                    exc,
                )
                await asyncio.sleep(1.0)
        return "", ""

    async def health(self) -> bool:
        """Best-effort keep-warm / readiness probe."""
        try:
            resp = await self._client.get(self._health_url, timeout=self._timeout_s)
            return resp.status_code == 200
        except Exception:  # noqa: BLE001
            return False

    async def aclose(self) -> None:
        await self._client.aclose()
