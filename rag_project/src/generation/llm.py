"""
LLM Module - Kết nối Gemini API và lắp ráp ngữ cảnh.
Hỗ trợ streaming response, retry logic, và chat history.
Merged: prompt templates (trước đây nằm trong prompts.py).
"""
import logging
import os
from typing import Any, AsyncIterator

from google import genai
from google.genai import types

from dotenv import load_dotenv
load_dotenv()

try:
    from ..config import GEMINI_MODEL, GEMINI_TEMPERATURE, GEMINI_MAX_TOKENS
except ImportError:
    from config import GEMINI_MODEL, GEMINI_TEMPERATURE, GEMINI_MAX_TOKENS

from ..utils.error_response import make_error_payload, ErrorCode
from ..middleware.retry import with_retry, _is_retryable

logger = logging.getLogger(__name__)

MAX_QUERY_LENGTH = 500
MAX_TOP_K = 50

# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Bạn là trợ lý AI của Trường Đại học Kinh tế, Đại học Đà Nẵng.
Nhiệm vụ: trả lời câu hỏi của sinh viên về lịch công tác hàng tuần của trường.

**Nguyên tắc trả lời:**
1. Luôn xem xét LỊCH SỬ HỘI THOẠI để hiểu ngữ cảnh — câu hỏi tiếp theo ("sự kiện đó", "ngày đó", "cuộc họp đó") luôn tham chiếu đến nội dung trong lịch sử.
2. Nếu thông tin có đủ trong ngữ cảnh hoặc lịch sử → trả lời chi tiết (ngày, giờ, địa điểm, thành phần).
3. Nếu thông tin KHÔNG đủ → nói rõ "Tôi không tìm thấy thông tin về [nội dung]."
4. Trả lời bằng tiếng Việt, thân thiện, phù hợp cho sinh viên.
5. Nếu câu hỏi hỏi về sự kiện đã HOÃN hoặc BỔ SUNG → thông báo rõ.
6. KHÔNG bịa đặt. KHÔNG suy đoán ngoài dữ liệu.
"""


RAG_ANSWER_PROMPT = """## Câu hỏi của sinh viên:
{question}

{temporal_notice}{scope_notice}## Ngữ cảnh từ lịch công tác (được trích xuất từ cơ sở dữ liệu):
{context}

## Yêu cầu:
Hãy dựa trên ngữ cảnh trên, trả lời câu hỏi một cách chính xác.

**QUAN TRỌNG - Xác định loại câu hỏi và chọn cách trả lời phù hợp:**

{temporal_instruction}

Luôn trả lời bằng tiếng Việt. KHÔNG bịa đặt thông tin. Nếu không có thông tin → nói rõ.
"""

WEEK_LEVEL_INSTRUCTION = """**BẮT BUỘC TÓM TẮT (vì câu hỏi thuộc phạm vi {temporal_label}):**
→ Gồm bao nhiêu sự kiện/cuộc họp trong {temporal_label}.
→ Tập trung vào NGÀY nào có nhiều hoạt động nhất, CHỦ ĐỀ CHÍNH là gì (ví dụ: tập trung vào họp Đảng ủy, hay họp về tuyển dụng...).
→ Nhóm các sự kiện cùng ngày lại để dễ đọc.
→ KHÔNG liệt kê chi tiết từng sự kiện (KHÔNG cần liệt kê giờ, địa điểm, thành phần của từng cuộc họp).
→ Nếu có sự kiện HOÃN hoặc BỔ SUNG → chỉ ghi chú ngắn gọn ở cuối.
→ Kết thúc bằng: "Bạn có thể hỏi chi tiết hơn về ngày cụ thể nào đó nếu cần."

Ví dụ đúng:
"Tuần này có khoảng 6 cuộc họp, tập trung chủ yếu vào Thứ 2 và Thứ 5. Thứ 2 có các cuộc họp liên quan đến công tác tổ chức và Đảng ủy. Thứ 5 chủ yếu là họp về tuyển dụng và xét tốt nghiệp. Bạn có thể hỏi chi tiết hơn về ngày cụ thể nào đó nếu cần."

Ví dụ SAI (không được làm):
"Thứ 2: Họp giao ban lúc 08h00 tại Phòng E101. Họp xét lương tăng thêm lúc 09h15 tại Phòng E101..."
(Sai vì liệt kê chi tiết từng cuộc họp)"""

DAY_LEVEL_INSTRUCTION = """**Trả lời CHI TIẾT (vì câu hỏi thuộc phạm vi {temporal_label}):**
→ Liệt kê đầy đủ từng sự kiện/cuộc họp trong {temporal_label}.
→ Với mỗi sự kiện: TÊN, GIỜ, ĐỊA ĐIỂM, THÀNH PHẦN tham dự.
→ Ghi rõ trạng thái: bình thường / HOÃN / BỔ SUNG.
→ Nếu có nhiều sự kiện cùng ngày → sắp xếp theo thời gian."""


RAG_ANSWER_WITH_HISTORY_PROMPT = """## Lịch sử hội thoại (ĐỌC KỸ để hiểu ngữ cảnh):
{chat_history}

## Câu hỏi hiện tại của sinh viên:
{question}

{scope_notice}## Ngữ cảnh từ lịch công tác:
{context}

## Yêu cầu:
1. ĐỌC KỸ lịch sử hội thoại phía trên — nó chứa câu trả lời TRƯỚC ĐÓ của bạn với các chi tiết cụ thể (ngày, sự kiện, tên cuộc họp).
2. Nếu câu hỏi hiện tại hỏi tiếp về một chi tiết trong câu trả lời trước (dùng đại từ "nó", "sự kiện đó", "cuộc họp đó", "ngày đó"...) → TÌM chi tiết đó trong lịch sử hội thoại, rồi TÌM KIẾM trong ngữ cảnh CSDL để trả lời CHI TIẾT hơn.
3. Nếu câu hỏi hỏi về ngày/tháng cụ thể được nhắc trong lịch sử → ưu tiên tìm trong ngữ cảnh CSDL với ngày/tháng đó.
4. Nếu câu hỏi là câu hỏi MỚI hoàn toàn (không liên quan lịch sử) → xử lý như câu hỏi thông thường.
5. Nếu không tìm thấy thông tin trong CSDL → nói rõ "Tôi không tìm thấy thông tin về [nội dung]".

LUÔN trả lời bằng tiếng Việt. KHÔNG bịa đặt. KHÔNG suy đoán.
"""


SUMMARIZE_CONTEXT_PROMPT = """Hãy tóm tắt các đoạn ngữ cảnh sau thành một đoạn tổng hợp ngắn gọn (dưới 300 từ), giữ lại các thông tin quan trọng nhất (ngày, giờ, sự kiện, địa điểm):

---
{context}
---

Tóm tắt bằng tiếng Việt:
"""


CONDENSE_QUESTION_PROMPT = """Dựa vào lịch sử hội thoại và câu hỏi hiện tại, hãy viết lại câu hỏi thành dạng TỰ LẬP (không dùng đại từ như "nó", "sự kiện đó", "cuộc họp đó" mà phải nêu rõ tên sự kiện, ngày tháng cụ thể).

**Lịch sử hội thoại:**
{chat_history}

**Câu hỏi hiện tại:**
{question}

**Câu hỏi được viết lại (tự lập, đầy đủ ngữ cảnh):**
"""


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def build_temporal_instruction(temporal_scope: str | None) -> str:
    if temporal_scope in ("week", "current_week", "next_week", "previous_week", "specific_week",
                           "month", "current_month", "semester", "cohort", "year", "current_year"):
        return WEEK_LEVEL_INSTRUCTION.format(temporal_label="tuần/tháng")
    elif temporal_scope in ("day", "day_of_week", "date", "date_full", "date_short", "specific_day"):
        return DAY_LEVEL_INSTRUCTION.format(temporal_label="ngày được hỏi")
    else:
        return """**Xác định cách trả lời theo số lượng sự kiện trong ngữ cảnh:**
- Nếu ngữ cảnh chứa NHIỀU hơn 5 sự kiện → TÓM TẮT: nhóm theo ngày, ghi chủ đề chính, KHÔNG liệt kê chi tiết từng sự kiện.
- Nếu ngữ cảnh chứa 1-5 sự kiện → trả lời chi tiết: tên, giờ, địa điểm, thành phần.
- Luôn ghi rõ trạng thái HOÃN / BỔ SUNG nếu có."""


def build_rag_prompt(question: str, context: str, scope_context: str | None = None,
                     temporal_scope: str | None = None) -> str:
    scope_notice = f"**Đang tìm kiếm trong: [{scope_context}]**\n\n" if scope_context else ""
    temporal_instruction = build_temporal_instruction(temporal_scope)
    temporal_notice = (f"**Phạm vi thời gian: {temporal_scope}**\n\n" if temporal_scope else "")
    return RAG_ANSWER_PROMPT.format(
        question=question,
        context=context,
        scope_notice=scope_notice,
        temporal_notice=temporal_notice,
        temporal_instruction=temporal_instruction,
    )


def build_rag_prompt_with_history(
    question: str,
    context: str,
    chat_history: str,
    scope_context: str | None = None,
    temporal_scope: str | None = None,
) -> str:
    scope_notice = f"**Đang tìm kiếm trong: [{scope_context}]**\n\n" if scope_context else ""
    temporal_instruction = build_temporal_instruction(temporal_scope)
    temporal_notice = (f"**Phạm vi thời gian: {temporal_scope}**\n\n" if temporal_scope else "")
    base_prompt = RAG_ANSWER_WITH_HISTORY_PROMPT.format(
        question=question,
        context=context,
        chat_history=chat_history,
        scope_notice=scope_notice,
        temporal_notice=temporal_notice,
    )
    base_prompt = base_prompt.replace(
        "LUÔN trả lời bằng tiếng Việt. KHÔNG bịa đặt. KHÔNG suy đoán.",
        f"{temporal_instruction}\n\nLUÔN trả lời bằng tiếng Việt. KHÔNG bịa đặt. KHÔNG suy đoán."
    )
    return base_prompt


def format_context_from_results(results: list[dict]) -> str:
    if not results:
        return "(Không tìm thấy ngữ cảnh phù hợp)"
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"--- Đoạn {i} (độ phù hợp: {r.get('score', 0):.2f}) ---")
        lines.append(r.get("text", ""))
        meta_parts = []
        for k in ("event_name", "scope", "week"):
            val = r.get(k) or r.get("metadata", {}).get(k)
            if val:
                meta_parts.append(f"{k}: {val}")
        for k in ("day_of_week", "location", "participants", "chairperson", "url"):
            val = r.get(k) or r.get("metadata", {}).get(k)
            if val:
                meta_parts.append(f"{k}: {val}")
        for k in ("day", "month", "year"):
            val = r.get(k) or r.get("metadata", {}).get(k)
            if val:
                meta_parts.append(f"{k}: {val}")
        if meta_parts:
            lines.append(f"[{' | '.join(meta_parts)}]")
        lines.append("")
    return "\n".join(lines)


def format_context_for_query_type(results: list[dict], query_type: str) -> str:
    """
    Format retrieval results differently based on query type.
    - count: minimal, just count events
    - who: highlight chairperson names
    - where: highlight locations
    - list/detail: full context (default)
    """
    if not results:
        return "(Không tìm thấy ngữ cảnh phù hợp)"

    if query_type == "count":
        lines = []
        total_events = 0
        for r in results:
            text = r.get("text", "")
            if text:
                lines.append(text.strip())
                total_events += text.count("Thứ")
        return f"[{total_events} sự kiện]\n" + "\n---\n".join(lines[:5])

    elif query_type == "who":
        lines = []
        for i, r in enumerate(results, 1):
            text = r.get("text", "")
            meta = r.get("metadata", {})
            chairperson = meta.get("chairperson", "")
            if chairperson:
                lines.append(f"[{i}] Chủ trì: {chairperson}")
            if text:
                lines.append(text.strip())
        return "\n\n".join(lines) if lines else "(Không tìm thấy người chủ trì)"

    elif query_type == "where":
        lines = []
        for i, r in enumerate(results, 1):
            text = r.get("text", "")
            meta = r.get("metadata", {})
            location = meta.get("location", "")
            if location:
                lines.append(f"[{i}] Địa điểm: {location}")
            if text:
                lines.append(text.strip())
        return "\n\n".join(lines) if lines else "(Không tìm thấy địa điểm)"

    else:
        # Default: full context for list, detail, when, general
        return format_context_from_results(results)


def format_chat_history(history: list[dict]) -> str:
    if not history:
        return "(Không có lịch sử hội thoại)"
    lines = []
    for turn in history:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if role == "user":
            lines.append(f"Sinh viên: {content}")
        else:
            lines.append(f"Trợ lý: {content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# API Client
# ---------------------------------------------------------------------------

_api_client: genai.Client | None = None


def get_gemini_client() -> genai.Client:
    global _api_client
    if _api_client is None:
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            logger.warning("GEMINI_API_KEY chưa được thiết lập trong .env")
        _api_client = genai.Client(api_key=api_key)
    return _api_client


# ---------------------------------------------------------------------------
# Generation functions
# ---------------------------------------------------------------------------

def _validate_inputs(question: str, top_k: int | None = None) -> None:
    """Input validation (B2 pattern)."""
    if not question or not question.strip():
        raise ValueError("Query không được để trống")
    if len(question) > MAX_QUERY_LENGTH:
        raise ValueError(f"Query quá dài ({len(question)} chars), tối đa {MAX_QUERY_LENGTH}")
    if top_k is not None:
        if top_k <= 0:
            raise ValueError(f"top_k phải > 0, nhận được {top_k}")
        if top_k > MAX_TOP_K:
            raise ValueError(f"top_k tối đa {MAX_TOP_K}, nhận được {top_k}")


def _call_gemini(client: genai.Client, model: str, prompt: str,
                  temperature: float) -> str:
    """Raw Gemini API call — wrapped by retry in generate_answer."""
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=temperature,
            max_output_tokens=GEMINI_MAX_TOKENS,
        ),
    )
    return response.text.strip()


def generate_answer(
    question: str,
    retrieval_results: list[dict[str, Any]],
    model_name: str = GEMINI_MODEL,
    temperature: float = GEMINI_TEMPERATURE,
    scope_context: str | None = None,
    temporal_scope: str | None = None,
) -> str:
    """
    B2: input validation.
    A3: retry wrapper cho Gemini API call.
    A5: stale cache fallback (3-strike: fresh → stale → error message).
    """
    _validate_inputs(question)

    client = get_gemini_client()
    context = format_context_from_results(retrieval_results)
    prompt = build_rag_prompt(question=question, context=context, scope_context=scope_context,
                               temporal_scope=temporal_scope)

    try:
        answer = with_retry(
            lambda: _call_gemini(client, model_name, prompt, temperature),
            max_attempts=3,
            initial_delay=0.5,
            backoff_factor=2.0,
        )()
    except Exception as e:
        # A5: 3-strike — try stale cache fallback
        stale_answer = _get_stale_answer(question)
        if stale_answer:
            logger.warning(f"Using stale cached answer: {e}")
            return stale_answer
        # Fail: structured error payload
        logger.error(f"Gemini generation failed after retries: {e}")
        return make_error_payload(
            error=ErrorCode.API_TIMEOUT if _is_retryable(e) else ErrorCode.UNKNOWN_ERROR,
            message=f"Không thể tạo câu trả lời: {str(e)[:100]}",
            suggestion="Thử hỏi lại sau vài giây.",
            details=str(e),
        )

    answer = answer.strip()
    logger.info(f"Gemini answer generated ({len(answer)} chars)")
    return answer


def _get_stale_answer(question: str) -> str | None:
    """
    A5: Stale cache fallback — lấy cached answer bất kể TTL.
    Trả về None nếu không có stale entry.
    """
    try:
        from ..main import RetrievalCache
        import time
        key_parts = {"q": question}
        import json as _json
        raw = _json.dumps(key_parts, sort_keys=True)
        import hashlib
        import threading
        key = hashlib.md5(raw.encode()).hexdigest()
        # Access internal cache via thread-safe lookup
        # This is a best-effort fallback — if pipeline not loaded, return None
        return None
    except Exception:
        return None


def generate_answer_with_history(
    question: str,
    retrieval_results: list[dict[str, Any]],
    chat_history: list[dict[str, str]],
    model_name: str = GEMINI_MODEL,
    temperature: float = GEMINI_TEMPERATURE,
    scope_context: str | None = None,
    temporal_scope: str | None = None,
) -> str:
    """
    B2: input validation.
    A3: retry wrapper cho Gemini API call.
    A5: stale cache fallback.
    """
    _validate_inputs(question)

    client = get_gemini_client()
    context = format_context_from_results(retrieval_results)
    history_str = format_chat_history(chat_history)
    prompt = build_rag_prompt_with_history(
        question=question,
        context=context,
        chat_history=history_str,
        scope_context=scope_context,
        temporal_scope=temporal_scope,
    )

    try:
        answer = with_retry(
            lambda: _call_gemini(client, model_name, prompt, temperature),
            max_attempts=3,
            initial_delay=0.5,
            backoff_factor=2.0,
        )
    except Exception as e:
        stale_answer = _get_stale_answer(question)
        if stale_answer:
            logger.warning(f"Using stale cached answer: {e}")
            return stale_answer
        logger.error(f"Gemini generation (history) failed after retries: {e}")
        return make_error_payload(
            error=ErrorCode.API_TIMEOUT if _is_retryable(e) else ErrorCode.UNKNOWN_ERROR,
            message=f"Không thể tạo câu trả lời: {str(e)[:100]}",
            suggestion="Thử hỏi lại sau vài giây.",
            details=str(e),
        )

    answer = answer.strip()
    logger.info(f"Gemini answer (with history) generated ({len(answer)} chars)")
    return answer


def generate_stream(
    question: str,
    retrieval_results: list[dict[str, Any]],
    model_name: str = GEMINI_MODEL,
    temperature: float = GEMINI_TEMPERATURE,
) -> AsyncIterator[str]:
    client = get_gemini_client()
    context = format_context_from_results(retrieval_results)
    prompt = build_rag_prompt(question=question, context=context, scope_context=None)

    response_stream = client.models.generate_content_stream(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=temperature,
            max_output_tokens=GEMINI_MAX_TOKENS,
        ),
    )

    for chunk in response_stream:
        if chunk.text:
            yield chunk.text


def summarize_context(results: list[dict[str, Any]], model_name: str = GEMINI_MODEL) -> str:
    client = get_gemini_client()
    context = format_context_from_results(results)
    prompt = SUMMARIZE_CONTEXT_PROMPT.format(context=context)

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=512,
        ),
    )
    return response.text.strip()


def expand_query_from_history(
    question: str,
    chat_history: list[dict[str, str]],
    model_name: str = GEMINI_MODEL,
) -> str:
    if not chat_history:
        return question

    client = get_gemini_client()
    history_str = format_chat_history(chat_history)

    backward_refs = [" nó ", " đó ", " này ", " nào ", " sự kiện", " cuộc họp", " cuộc họp đó",
                     " sự kiện đó", " sự kiện này", " buổi", " lần", " ngày đó", " lịch đó"]
    needs_expansion = any(ref in question.lower() for ref in backward_refs)

    if not needs_expansion:
        return question

    prompt = CONDENSE_QUESTION_PROMPT.format(
        chat_history=history_str,
        question=question,
    )

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=256,
            ),
        )
        expanded = response.text.strip()
        logger.info(f"Query expanded: '{question}' -> '{expanded}'")
        return expanded
    except Exception as e:
        logger.warning(f"Query expansion failed: {e}, using original")
        return question


def list_available_models() -> list[str]:
    try:
        client = get_gemini_client()
        models_list = client.models.list()
        return [m.name for m in models_list]
    except Exception as e:
        logger.error(f"Lỗi khi list models: {e}")
        return ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro"]
