"""
QueryParser — LLM-based query parsing với structured JSON output.
Thay thế: temporal.py, content_filter.py, query_intent.py (regex-based).

Design:
- Single Gemini call per query, structured JSON output
- Few-shot prompt cho tất cả 7 loại query trong benchmark
- Graceful fallback: LLM failure → regex-based fallback → default
- Confidence scoring: high when LLM succeeds, low on fallback
"""
import json
import logging
import os
import re
import time
from datetime import datetime

from google import genai
from google.genai import types

from .schemas import ParsedQuery, TemporalSpec, ContentSpec, QueryType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt Template — few-shot cho Gemini structured output
# ---------------------------------------------------------------------------

_QUERY_PARSE_PROMPT = """Bạn là một query parser cho hệ thống hỏi đáp lịch công tác Đại học Kinh tế Đà Nẵng.
Nhiệm vụ: Phân tích câu hỏi và trả về JSON mô tả cấu trúc của nó.

QUY TẮC QUAN TRỌNG:
- Phân tích TIẾNG VIỆT tự nhiên
- Trả về JSON hợp lệ, không có markdown, không có giải thích
- Nếu không detect được temporal → has_reference = false
- Đối với "năm vừa rồi" → year = 2025 (vì dữ liệu hiện tại là 2025)
- Đối với "tháng trước" → dùng "previous_month" làm type
- Đối với "tuần tới" → relative = "next_week"
- Đối với "tuần này" → relative = "current_week"
- Đối với "tháng này" → type = "current_month"
- Đối với chairperson: nhận diện chức vụ như "hiệu trưởng", "phó hiệu trưởng", "trưởng phòng", v.v.
- query_type: "list" = liệt kê sự kiện, "count" = đếm, "who" = hỏi người, "when" = hỏi thời gian, "where" = hỏi địa điểm, "detail" = chi tiết

VÍ DỤ:

Câu hỏi: "Lịch làm việc của hiệu trưởng trong năm vừa rồi"
JSON: {{"has_reference": true, "type": "year", "year": 2025, "month": null, "day": null, "day_of_week": null, "week": null, "relative": null, "chairperson": "Hiệu trưởng", "event_name": null, "participants": null, "is_list_query": true, "query_type": "list"}}

Câu hỏi: "Tuần này có sự kiện gì"
JSON: {{"has_reference": true, "type": "current_week", "year": null, "month": null, "day": null, "day_of_week": null, "week": null, "relative": "current_week", "chairperson": null, "event_name": null, "participants": null, "is_list_query": true, "query_type": "list"}}

Câu hỏi: "Tháng 3 có cuộc học gì"
JSON: {{"has_reference": true, "type": "month", "year": null, "month": 3, "day": null, "day_of_week": null, "week": null, "relative": null, "chairperson": null, "event_name": "học", "participants": null, "is_list_query": true, "query_type": "list"}}

Câu hỏi: "Họp giao ban do ai chủ trì"
JSON: {{"has_reference": false, "type": "general", "year": null, "month": null, "day": null, "day_of_week": null, "week": null, "relative": null, "chairperson": null, "event_name": "giao ban", "participants": null, "is_list_query": false, "query_type": "who"}}

Câu hỏi: "Ai tham gia họp phòng ban thứ 6"
JSON: {{"has_reference": true, "type": "day_of_week", "year": null, "month": null, "day": null, "day_of_week": "Thứ Sáu", "week": null, "relative": null, "chairperson": null, "event_name": "họp", "participants": "phòng ban", "is_list_query": false, "query_type": "who"}}

Câu hỏi: "Lịch làm việc tháng trước"
JSON: {{"has_reference": true, "type": "previous_month", "year": null, "month": null, "day": null, "day_of_week": null, "week": null, "relative": null, "chairperson": null, "event_name": null, "participants": null, "is_list_query": true, "query_type": "list"}}

Câu hỏi: "Tuần tới có hoạt động gì"
JSON: {{"has_reference": true, "type": "next_week", "year": null, "month": null, "day": null, "day_of_week": null, "week": null, "relative": "next_week", "chairperson": null, "event_name": null, "participants": null, "is_list_query": true, "query_type": "list"}}

Câu hỏi: "Ai chủ trì họp tuyển dụng tháng 5"
JSON: {{"has_reference": true, "type": "month", "year": null, "month": 5, "day": null, "day_of_week": null, "week": null, "relative": null, "chairperson": null, "event_name": "tuyển dụng", "participants": null, "is_list_query": false, "query_type": "who"}}

Câu hỏi: "Thứ 2 tuần này có lịch gì"
JSON: {{"has_reference": true, "type": "current_week", "year": null, "month": null, "day": null, "day_of_week": "Thứ Hai", "week": null, "relative": "current_week", "chairperson": null, "event_name": null, "participants": null, "is_list_query": true, "query_type": "list"}}

Câu hỏi: "Những hoạt động nào diễn ra vào thứ 6"
JSON: {{"has_reference": true, "type": "day_of_week", "year": null, "month": null, "day": null, "day_of_week": "Thứ Sáu", "week": null, "relative": null, "chairperson": null, "event_name": null, "participants": null, "is_list_query": true, "query_type": "list"}}

Câu hỏi: "Có bao nhiêu sự kiện tuần này"
JSON: {{"has_reference": true, "type": "current_week", "year": null, "month": null, "day": null, "day_of_week": null, "week": null, "relative": "current_week", "chairperson": null, "event_name": null, "participants": null, "is_list_query": true, "query_type": "count"}}

Câu hỏi: "Sự kiện nào vào ngày 15 tháng 5"
JSON: {{"has_reference": true, "type": "date", "year": null, "month": 5, "day": 15, "day_of_week": null, "week": null, "relative": null, "chairperson": null, "event_name": null, "participants": null, "is_list_query": true, "query_type": "list"}}

Câu hỏi: "Địa điểm tổ chức các sự kiện tuần này"
JSON: {{"has_reference": true, "type": "current_week", "year": null, "month": null, "day": null, "day_of_week": null, "week": null, "relative": "current_week", "chairperson": null, "event_name": null, "participants": null, "is_list_query": true, "query_type": "where"}}

Câu hỏi: "Phó hiệu trưởng có lịch họp gì tuần tới"
JSON: {{"has_reference": true, "type": "next_week", "year": null, "month": null, "day": null, "day_of_week": null, "week": null, "relative": "next_week", "chairperson": "Phó Hiệu trưởng", "event_name": null, "participants": null, "is_list_query": true, "query_type": "list"}}

Câu hỏi: "Lịch năm 2025"
JSON: {{"has_reference": true, "type": "year", "year": 2025, "month": null, "day": null, "day_of_week": null, "week": null, "relative": null, "chairperson": null, "event_name": null, "participants": null, "is_list_query": true, "query_type": "list"}}

Câu hỏi: "Tháng này có đào tạo gì không"
JSON: {{"has_reference": true, "type": "current_month", "year": null, "month": null, "day": null, "day_of_week": null, "week": null, "relative": null, "chairperson": null, "event_name": "đào tạo", "participants": null, "is_list_query": true, "query_type": "list"}}

Câu hỏi: "Lịch công tác nội bộ tuần 40"
JSON: {{"has_reference": true, "type": "specific_week", "year": null, "month": null, "day": null, "day_of_week": null, "week": "Tuần 40", "relative": null, "chairperson": null, "event_name": null, "participants": null, "is_list_query": true, "query_type": "list"}}

Câu hỏi: "Cho tôi xem các cuộc họp trong tháng"
JSON: {{"has_reference": true, "type": "current_month", "year": null, "month": null, "day": null, "day_of_week": null, "week": null, "relative": null, "chairperson": null, "event_name": "họp", "participants": null, "is_list_query": true, "query_type": "list"}}

Câu hỏi: "Họp giao ban phòng ban thứ 6 do ông nào chủ trì"
JSON: {{"has_reference": true, "type": "day_of_week", "year": null, "month": null, "day": null, "day_of_week": "Thứ Sáu", "week": null, "relative": null, "chairperson": null, "event_name": "giao ban", "participants": "phòng ban", "is_list_query": false, "query_type": "who"}}

Câu hỏi: "Trường có hoạt động gì trong học kỳ 2"
JSON: {{"has_reference": true, "type": "semester", "year": null, "month": null, "day": null, "day_of_week": null, "week": null, "relative": null, "chairperson": null, "event_name": null, "participants": null, "is_list_query": true, "query_type": "list"}}

---

Bây giờ phân tích câu hỏi sau và trả về JSON:
Câu hỏi: "{query}"
JSON: """


# ---------------------------------------------------------------------------
# Gemini Client (singleton)
# ---------------------------------------------------------------------------

_gemini_client: genai.Client | None = None


def _get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            logger.warning("GEMINI_API_KEY chưa được thiết lập")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


# ---------------------------------------------------------------------------
# QueryParser
# ---------------------------------------------------------------------------

class QueryParser:
    """
    LLM-based query parser — thay thế regex-based temporal.py,
    content_filter.py, query_intent.py.

    parse() returns ParsedQuery (Pydantic) — dùng cho:
    1. Benchmark shim (Step 0): đo baseline
    2. Production: QueryParser được gắn vào RAGPipeline
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.0,
    ):
        self.model = model
        self.temperature = temperature
        self._client: genai.Client | None = None
        self._llm_call_count = 0
        self._llm_error_count = 0

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            self._client = _get_gemini_client()
        return self._client

    def parse(self, query: str) -> ParsedQuery:
        """
        Parse một câu hỏi → ParsedQuery.
        Luôn trả về ParsedQuery (không raise exception).
        """
        try:
            return self._parse_with_llm(query)
        except Exception as e:
            logger.warning(f"QueryParser LLM failed: {e}")
            self._llm_error_count += 1
            return self._parse_with_regex_fallback(query)

    def _parse_with_llm(self, query: str) -> ParsedQuery:
        """Gọi Gemini để parse, retry 1 lần nếu quota."""
        prompt = _QUERY_PARSE_PROMPT.format(query=query)

        last_error: Exception | None = None
        for attempt in range(2):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=self.temperature,
                        max_output_tokens=256,
                    ),
                )
                text = response.text.strip()

                # Extract JSON from response
                raw = text
                # Handle markdown code blocks
                if raw.startswith("```"):
                    lines = raw.split("\n")
                    raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

                data = json.loads(raw)
                self._llm_call_count += 1
                return self._build_parsed_query(data, query)

            except (json.JSONDecodeError, KeyError, TypeError) as e:
                last_error = e
                logger.warning(f"QueryParser JSON parse failed (attempt {attempt + 1}): {e}")
                continue
            except Exception as e:
                err_str = str(e)
                last_error = e
                if "429" in err_str:
                    logger.warning(f"QueryParser 429 quota — falling back to regex immediately")
                    raise  # → catch in parse() → regex fallback
                raise

        # All retries failed → fallback
        raise last_error or RuntimeError("QueryParser LLM failed after retries")

    def _build_parsed_query(self, data: dict, query: str) -> ParsedQuery:
        """Build ParsedQuery từ JSON dict của LLM."""
        has_ref = data.get("has_reference", False)

        # Determine temporal type
        temporal_type = data.get("type", "general")

        # Resolve relative temporal references
        relative = data.get("relative")
        if relative in ("next_week", "previous_week", "current_week"):
            temporal_type = relative

        # Resolve previous_month
        if temporal_type == "previous_month":
            has_ref = True
            # Don't set specific month — let the retrieval layer resolve it
            # based on the current date

        # Resolve current_month
        if temporal_type == "current_month":
            has_ref = True

        temporal = TemporalSpec(
            has_reference=has_ref,
            type=temporal_type,
            year=data.get("year"),
            month=data.get("month"),
            day=data.get("day"),
            day_of_week=data.get("day_of_week"),
            week=data.get("week"),
            relative=relative,
        )

        content = ContentSpec(
            is_list_query=data.get("is_list_query", True),
            chairperson=data.get("chairperson"),
            event_name=data.get("event_name"),
            participants=data.get("participants"),
        )

        query_type_str = data.get("query_type", "list")
        try:
            query_type = QueryType(query_type_str)
        except ValueError:
            query_type = QueryType.LIST

        is_general = not has_ref

        return ParsedQuery(
            temporal=temporal,
            content=content,
            query_type=query_type,
            is_general_query=is_general,
            confidence=0.95,
        )

    def _parse_with_regex_fallback(self, query: str) -> ParsedQuery:
        """
        Fallback: regex-based parsing khi LLM fail.
        Đây là logic từ temporal.py + content_filter.py gộp lại.
        """
        q = query.lower()
        temporal = TemporalSpec()
        content = ContentSpec()
        confidence = 0.3  # Low confidence for fallback

        # --- Temporal patterns ---
        # Year
        year_match = re.search(r"năm\s*(\d{4})", q)
        if year_match:
            temporal.year = int(year_match.group(1))
            temporal.has_reference = True
            temporal.type = "year"
        elif "năm vừa rồi" in q or "năm ngoái" in q:
            temporal.year = 2025
            temporal.has_reference = True
            temporal.type = "year"

        # Month (explicit number)
        month_match = re.search(r"tháng\s*(\d{1,2})", q)
        if month_match:
            temporal.month = int(month_match.group(1))
            temporal.has_reference = True
            temporal.type = "month"

        # Day of week
        dow_map = {
            "thứ hai": "Thứ Hai", "thứ 2": "Thứ Hai",
            "thứ ba": "Thứ Ba", "thứ 3": "Thứ Ba",
            "thứ tư": "Thứ Tư", "thứ 4": "Thứ Tư",
            "thứ năm": "Thứ Năm", "thứ 5": "Thứ Năm",
            "thứ sáu": "Thứ Sáu", "thứ 6": "Thứ Sáu",
            "thứ bảy": "Thứ Bảy", "thứ 7": "Thứ Bảy",
            "chủ nhật": "Chủ Nhật", "cn": "Chủ Nhật",
        }
        for pattern, dow in dow_map.items():
            if pattern in q:
                temporal.day_of_week = dow
                temporal.has_reference = True
                temporal.type = "day_of_week"
                break

        # Relative temporal
        if "tuần tới" in q or "tuần sau" in q:
            temporal.relative = "next_week"
            temporal.has_reference = True
            temporal.type = "next_week"
        elif "tuần này" in q:
            temporal.relative = "current_week"
            temporal.has_reference = True
            temporal.type = "current_week"
        elif "tuần trước" in q or "tuần qua" in q:
            temporal.relative = "previous_week"
            temporal.has_reference = True
            temporal.type = "previous_week"
        elif "tháng trước" in q or "tháng qua" in q:
            temporal.type = "previous_month"
            temporal.has_reference = True
        elif "tháng này" in q:
            temporal.type = "current_month"
            temporal.has_reference = True

        # Specific week
        week_match = re.search(r"tuần\s*(\d+)", q)
        if week_match:
            wn = int(week_match.group(1))
            temporal.week = f"Tuần {wn}"
            temporal.has_reference = True
            temporal.type = "specific_week"

        # Date (day + month)
        date_match = re.search(r"ngày\s*(\d{1,2})\s*tháng\s*(\d{1,2})", q)
        if date_match:
            temporal.day = int(date_match.group(1))
            temporal.month = int(date_match.group(2))
            temporal.has_reference = True
            temporal.type = "date"

        # Semester
        if "học kỳ" in q:
            temporal.type = "semester"
            temporal.has_reference = True

        # --- Content patterns ---
        # Chairperson
        if "hiệu trưởng" in q:
            content.chairperson = "Hiệu trưởng"
        elif "phó hiệu trưởng" in q:
            content.chairperson = "Phó Hiệu trưởng"
        elif "trưởng phòng" in q:
            content.chairperson = "Trưởng phòng"

        # Event name (simple keyword extraction)
        event_keywords = ["giao ban", "tuyển dụng", "đào tạo", "họp", "seminar", "thi"]
        for kw in event_keywords:
            if kw in q:
                content.event_name = kw
                break

        # --- Query type ---
        if any(w in q for w in ["bao nhieu", "bao nhiêu", "dem", "số lượng"]):
            query_type = QueryType.COUNT
        elif any(w in q for w in ["ai", "chu tri", "tham gia"]):
            query_type = QueryType.WHO
        elif any(w in q for w in ["o dau", "địa điểm", "tai cho nao"]):
            query_type = QueryType.WHERE
        elif any(w in q for w in ["khi nao", "ngay nao", "bao gio"]):
            query_type = QueryType.WHEN
        elif any(w in q for w in ["chi tiet", "cụ thể"]):
            query_type = QueryType.DETAIL
        else:
            query_type = QueryType.LIST

        return ParsedQuery(
            temporal=temporal,
            content=content,
            query_type=query_type,
            is_general_query=not temporal.has_reference,
            confidence=confidence,
        )
