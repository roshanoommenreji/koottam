"""One async client for every server. All providers speak the OpenAI-compatible
/chat/completions API, so Cerebras, OpenRouter, Groq and the laptop's Ollama differ
only in base_url, key and how fast we are allowed to call them."""

import asyncio
import time

import httpx

from koottam.config import ModelConfig

RETRYABLE = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 6


class RateLimiter:
    """Spaces request *starts* at least 60/rpm seconds apart. Simpler than a token bucket
    and enough here: free tiers are the bottleneck, not our concurrency."""

    def __init__(self, rpm: int) -> None:
        self.interval = 60.0 / max(rpm, 1)
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            delay = self._next - now
            self._next = max(now, self._next) + self.interval
        if delay > 0:
            await asyncio.sleep(delay)


class ChatClient:
    def __init__(self, cfg: ModelConfig, http: httpx.AsyncClient) -> None:
        self.cfg = cfg
        self.http = http
        self.limiter = RateLimiter(cfg.rpm)

    async def chat(
        self, messages: list[dict[str, str]], max_tokens: int = 2048, temperature: float = 0.0
    ) -> str:
        """The reply text. Retries rate limits and server errors; raises on anything else
        (bad key, unknown model), because retrying those just wastes the quota."""
        headers = {"Authorization": f"Bearer {self.cfg.api_key}"} if self.cfg.api_key else {}
        body = {
            "model": self.cfg.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **self.cfg.extra,
        }
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        for attempt in range(MAX_ATTEMPTS):
            await self.limiter.wait()
            try:
                resp = await self.http.post(url, json=body, headers=headers)
            except httpx.TransportError:
                if attempt == MAX_ATTEMPTS - 1:
                    raise
                await asyncio.sleep(2**attempt)
                continue
            if resp.status_code in RETRYABLE and attempt < MAX_ATTEMPTS - 1:
                await asyncio.sleep(_backoff(resp, attempt))
                continue
            resp.raise_for_status()
            message = resp.json()["choices"][0]["message"]
            # Reasoning models can spend the whole budget thinking and return no content.
            return str(message.get("content") or "")
        raise RuntimeError("unreachable")


def _backoff(resp: httpx.Response, attempt: int) -> float:
    retry_after = resp.headers.get("retry-after")
    if retry_after:
        try:
            return min(float(retry_after), 300.0)
        except ValueError:
            pass
    return float(min(2 ** (attempt + 1), 60))
