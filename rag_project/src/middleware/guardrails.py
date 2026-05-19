"""
guardrails.py — LLM-based content filter trước khi xử lý query.
Pattern: fail-open, allow-list check trước LLM classify,
structured rejection với Gemini 2.5 Flash.
"""
import logging
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class GuardrailsDecision(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    ERROR = "error"  # classification error → fail-open


ALLOW_LIST = {
    "lịch", "sự kiện", "họp", "cuộc họp", "làm việc",
    "phòng", "ban", "hiệu trưởng", "phó hiệu trưởng",
    "tháng", "tuần", "ngày", "thứ", "năm", "học kỳ",
    "chủ trì", "tham gia", "địa điểm", "đào tạo", "tuyển dụng",
    "sinh viên", "giảng viên", "công tác", "đảng", "hội nghị",
    "xét", "thi", "bảo vệ", "seminar", "workshop",
    "họp giao ban", "phó hiệu trưởng", "thanh tra",
}

BLOCK_LIST = {
    "hack", "exploit", "inject", "sql injection", "xss",
    "phishing", "malware", "virus", "bomb",
}


class GuardrailsMiddleware:
    """LLM-based content filter cho RAG queries."""

    def __init__(self, fail_open: bool = True):
        self.fail_open = fail_open

    def _quick_allow_check(self, query: str) -> Optional[GuardrailsDecision]:
        """Fast allow/block check trước khi gọi LLM."""
        q_lower = query.lower()

        for blocked in BLOCK_LIST:
            if blocked in q_lower:
                return GuardrailsDecision.BLOCK

        for allowed in ALLOW_LIST:
            if allowed in q_lower:
                return GuardrailsDecision.ALLOW

        return None  # indeterminate → LLM classify

    def _classify_with_llm(self, query: str) -> GuardrailsDecision:
        """Dùng Gemini để classify query."""
        try:
            from google import genai
            import os
            from dotenv import load_dotenv
            load_dotenv()

            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))
            prompt = f"""Bạn là guardrails classifier cho hệ thống hỏi đáp lịch công tác trường ĐH Kinh tế Đà Nẵng.

Query: "{query}"

Chỉ trả lời một từ: ALLOW nếu query hỏi về lịch công tác, sự kiện, cuộc họp, nhân sự, thời gian, địa điểm.
BLOCK nếu query chứa nội dung độc hại, yêu cầu bất thường, hoặc không liên quan đến lịch công tác.
"""
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={"temperature": 0.0, "max_output_tokens": 8},
            )
            decision = response.text.strip().upper()
            if decision.startswith("BLOCK"):
                return GuardrailsDecision.BLOCK
            return GuardrailsDecision.ALLOW
        except Exception as e:
            logger.warning(f"Guardrails LLM classify failed: {e}")
            return GuardrailsDecision.ERROR

    def check(self, query: str) -> GuardrailsDecision:
        """
        Kiểm tra query và trả về GuardrailsDecision.
        Fail-open: nếu classification error → ALLOW.
        """
        if not query or not query.strip():
            return GuardrailsDecision.ERROR

        quick = self._quick_allow_check(query)
        if quick is not None:
            return quick

        decision = self._classify_with_llm(query)
        if decision == GuardrailsDecision.ERROR:
            return GuardrailsDecision.ALLOW if self.fail_open else GuardrailsDecision.BLOCK

        return decision

    def get_block_message(self) -> str:
        return (
            "Xin lỗi, câu hỏi của bạn không thể được xử lý "
            "vì nội dung không liên quan đến lịch công tác trường ĐH Kinh tế Đà Nẵng. "
            "Bạn có thể hỏi về lịch làm việc, cuộc họp, sự kiện, nhân sự, thời gian, địa điểm..."
        )


_guardrails: Optional[GuardrailsMiddleware] = None


def get_guardrails() -> GuardrailsMiddleware:
    global _guardrails
    if _guardrails is None:
        _guardrails = GuardrailsMiddleware()
    return _guardrails
