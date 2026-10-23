"""
Bounded async request queue with concurrency control.

Prevents the CPU-heavy TTS model from being hammered by concurrent requests,
which would cause severe latency spikes on a 4-core laptop CPU.

Settings (from .env):
  TTS_MAX_CONCURRENT   – max simultaneous synthesis jobs (default: 1)
  TTS_QUEUE_SIZE       – max queued requests beyond concurrent (default: 4)
  TTS_REQUEST_TIMEOUT  – seconds before queued request is rejected (default: 30)
"""

import asyncio
import os
import time
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

logger = logging.getLogger(__name__)

_MAX_CONCURRENT = int(os.environ.get("TTS_MAX_CONCURRENT", "1"))
_QUEUE_SIZE = int(os.environ.get("TTS_QUEUE_SIZE", "4"))
_REQUEST_TIMEOUT = float(os.environ.get("TTS_REQUEST_TIMEOUT", "30"))

# Semaphore limits simultaneous synthesis calls
_semaphore: asyncio.Semaphore = asyncio.Semaphore(1)   # re-init in setup()
_queue_depth: int = 0                                    # current waiting count


def setup() -> None:
    """
    Initialise the semaphore with the configured concurrency.
    Call this once at startup inside the lifespan context.
    """
    global _semaphore
    _semaphore = asyncio.Semaphore(_MAX_CONCURRENT)
    logger.info(
        "Queue manager: max_concurrent=%d queue_size=%d timeout=%.1fs",
        _MAX_CONCURRENT, _QUEUE_SIZE, _REQUEST_TIMEOUT,
    )


@asynccontextmanager
async def synthesis_slot() -> AsyncGenerator[None, None]:
    """
    Async context manager that gates access to the TTS model.

    Raises:
        asyncio.TimeoutError  → caller should return HTTP 503
        RuntimeError          → queue full, caller should return HTTP 429
    """
    global _queue_depth

    if _queue_depth >= _QUEUE_SIZE:
        raise RuntimeError(
            f"Request queue full ({_queue_depth}/{_QUEUE_SIZE}). "
            "Server is overloaded, please retry later."
        )

    _queue_depth += 1
    enqueue_time = time.monotonic()
    try:
        # Wait for a concurrency slot, with timeout
        await asyncio.wait_for(_semaphore.acquire(), timeout=_REQUEST_TIMEOUT)
        wait_ms = (time.monotonic() - enqueue_time) * 1000
        if wait_ms > 100:
            logger.info("Request waited %.0f ms in queue", wait_ms)
        try:
            yield
        finally:
            _semaphore.release()
    finally:
        _queue_depth -= 1


def current_queue_depth() -> int:
    return _queue_depth
