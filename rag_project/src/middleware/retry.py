"""
retry.py — Exponential backoff retry với jitter.
Pattern: retryable status codes {429, 500, 502, 503, 504}, max_attempts=3,
delay=0.5s, factor=2.0, fail-gracefully.
"""
import asyncio
import logging
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
RETRYABLE_ERROR_MARKERS = (
    "bad gateway",
    "connection error",
    "connection reset",
    "gateway time-out",
    "gateway timeout",
    "service unavailable",
    "temporarily unavailable",
    "timed out",
    "timeout",
    "too many requests",
    "rate limit",
)

T = TypeVar("T")


def _is_retryable(error: Exception) -> bool:
    status = getattr(error, "status_code", None)
    if isinstance(status, int) and status in RETRYABLE_STATUS_CODES:
        return True
    resp = getattr(error, "response", None)
    resp_status = getattr(resp, "status_code", None) if resp else None
    if isinstance(resp_status, int) and resp_status in RETRYABLE_STATUS_CODES:
        return True
    text = str(error).lower()
    return any(m in text for m in RETRYABLE_ERROR_MARKERS)


def with_retry(
    func: Callable[..., T],
    *,
    max_attempts: int = 3,
    initial_delay: float = 0.5,
    backoff_factor: float = 2.0,
) -> Callable[..., T]:
    """Decorator wrap function với exponential backoff retry."""
    def wrapper(*args, **kwargs) -> T:
        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if _is_retryable(e) and attempt < max_attempts:
                    delay = initial_delay * (backoff_factor ** (attempt - 1))
                    logger.warning(
                        f"Retryable error (attempt {attempt}/{max_attempts}): {e}, "
                        f"waiting {delay:.1f}s"
                    )
                    time.sleep(delay)
                    continue
                logger.error(f"Non-retryable or exhausted: {e}")
                raise
        raise last_error
    return wrapper


async def with_retry_async(
    coro_fn: Callable[..., Any],
    *,
    max_attempts: int = 3,
    initial_delay: float = 0.5,
    backoff_factor: float = 2.0,
) -> Any:
    """Async version của exponential backoff retry."""
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await coro_fn()
        except Exception as e:
            last_error = e
            if _is_retryable(e) and attempt < max_attempts:
                delay = initial_delay * (backoff_factor ** (attempt - 1))
                logger.warning(
                    f"Async retryable error (attempt {attempt}/{max_attempts}): {e}, "
                    f"waiting {delay:.1f}s"
                )
                await asyncio.sleep(delay)
                continue
            logger.error(f"Non-retryable or exhausted: {e}")
            raise
    raise last_error
