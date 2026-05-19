"""
error_response.py — Structured JSON error payloads thay vì raise exception.
Pattern: fail-gracefully, structured error thay vì raw exceptions.
"""
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ErrorCode:
    API_TIMEOUT = "API_TIMEOUT"
    API_RATE_LIMIT = "API_RATE_LIMIT"
    API_SERVER_ERROR = "API_SERVER_ERROR"
    API_BAD_REQUEST = "API_BAD_REQUEST"
    API_UNAUTHORIZED = "API_UNAUTHORIZED"
    PARSE_ERROR = "PARSE_ERROR"
    RETRIEVAL_ERROR = "RETRIEVAL_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    GUARDRAILS_BLOCKED = "GUARDRAILS_BLOCKED"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


def make_error_payload(
    error: str,
    message: str,
    suggestion: str,
    details: str = "",
) -> str:
    """Trả về JSON error payload thay vì raise exception."""
    payload = {
        "error": error,
        "message": message,
        "suggestion": suggestion,
        "details": details[:200] if details else "",
    }
    logger.warning(f"Error payload: {error} — {message}")
    return json.dumps(payload, ensure_ascii=False)


def parse_error_payload(raw: str) -> Optional[dict]:
    """Parse JSON error payload, trả về dict hoặc None."""
    try:
        return json.loads(raw)
    except Exception:
        return None


def is_error_payload(value: str) -> bool:
    """Kiểm tra xem string có phải là error payload không."""
    return parse_error_payload(value) is not None
