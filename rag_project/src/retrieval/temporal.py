"""
Temporal Search Module - Utility functions for temporal filter resolution.
Regex-based temporal detection is DEPRECATED — use QueryParser (LLM) instead.
This module contains only resolve/boost utilities used by hybrid_search.
"""
import calendar
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


def _get_latest_week_from_db(available_weeks: list[str] | None) -> str | None:
    """
    Trả về tuần mới nhất trong database (theo số tuần lớn nhất).
    Đây là 'tuần này' trong ngữ cảnh lịch — không phải ISO week thực.
    """
    if not available_weeks:
        return None
    nums = [extract_week_number(w) for w in available_weeks if extract_week_number(w)]
    if not nums:
        return None
    latest = max(nums)
    return f"Tuần {latest:02d}"


@dataclass
class TemporalIntent:
    """Kết quả phân tích ý định thời gian từ câu hỏi."""
    has_temporal_reference: bool = False
    explicit_week: str | None = None
    explicit_week_num: int | None = None
    explicit_day_of_week: str | None = None
    explicit_day_of_week_num: int | None = None
    explicit_date: str | None = None
    explicit_day: int | None = None
    explicit_day_month: int | None = None
    explicit_day_year: int | None = None
    explicit_month: int | None = None
    explicit_year: int | None = None
    explicit_location: str | None = None
    temporal_type: str | None = None
    recency_boost_needed: bool = False


@dataclass
class TemporalSearchConfig:
    """Cấu hình temporal search."""
    recency_boost_weight: float = 0.15
    enable_recency_boost: bool = True
    max_boost_factor: float = 2.0


# ---------------------------------------------------------------------------
# Week Utilities
# ---------------------------------------------------------------------------

def extract_week_number(week_field: str | None) -> int | None:
    """Trích xuất số tuần từ trường week (e.g. 'Tuần 5' → 5)."""
    if not week_field:
        return None
    m = re.search(r"W?(\d+)", week_field, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def parse_week_field(week_field: str | None) -> dict[str, Any]:
    """
    Parse trường week từ metadata, trả về dict có keys: raw, week_num, order_key.
    order_key dùng để sort tuần gần nhất lên đầu.
    """
    if not week_field:
        return {"raw": None, "week_num": None, "order_key": 0}

    week_num = extract_week_number(week_field)
    if week_num is not None:
        return {"raw": week_field, "week_num": week_num, "order_key": week_num}
    return {"raw": week_field, "week_num": None, "order_key": 0}


def _nearest_week(target_num: int, available_weeks: list[str]) -> str:
    """Trả về tuần gần nhất với target_num trong available_weeks."""
    nums = [extract_week_number(w) for w in available_weeks if extract_week_number(w)]
    if not nums:
        return f"Tuần {target_num:02d}"
    nearest = min(nums, key=lambda n: abs(n - target_num))
    return f"Tuần {nearest:02d}"


# ---------------------------------------------------------------------------
# Recency Boost
# ---------------------------------------------------------------------------

def build_recency_boost(
    results: list[dict[str, Any]],
    config: TemporalSearchConfig | None = None,
) -> list[dict[str, Any]]:
    """
    Áp recency boost cho kết quả khi user không đề cập thời gian cụ thể.
    Tuần gần nhất (số tuần lớn nhất) nhận boost cao nhất.
    Tuần cũ hơn nhận boost giảm dần theo khoảng cách tuần.
    """
    if config is None:
        config = TemporalSearchConfig()

    if not config.enable_recency_boost or not results:
        return results

    week_nums = []
    for r in results:
        parsed = parse_week_field(r.get("week", ""))
        week_nums.append(parsed["week_num"])

    valid_weeks = [w for w in week_nums if w is not None]
    if not valid_weeks:
        return results

    max_week = max(valid_weeks)
    min_week = min(valid_weeks)
    week_range = max(max_week - min_week, 1)

    boosted = []
    for i, r in enumerate(results):
        wn = week_nums[i]
        if wn is not None:
            distance = max_week - wn
            factor = config.max_boost_factor - (distance / week_range) * (config.max_boost_factor - 1.0)
            factor = max(1.0, factor)
        else:
            factor = 1.0

        original_score = r.get("score", 0)
        boosted_score = original_score * factor

        boosted.append({
            **r,
            "score": round(boosted_score, 4),
            "recency_factor": round(factor, 3),
            "week_num": wn,
        })

    boosted.sort(key=lambda x: x["score"], reverse=True)
    return boosted


# ---------------------------------------------------------------------------
# Temporal Filter Resolution
# ---------------------------------------------------------------------------

def resolve_week_filter(
    intent: TemporalIntent,
    available_weeks: list[str] | None = None,
) -> str | None:
    """
    Chuyển TemporalIntent thành giá trị filter cho Qdrant.
    'current'/'next'/'previous' được tính từ tuần mới nhất TRONG DATABASE,
    KHÔNG dùng ISO week thực từ hệ thống.
    Nếu tuần target không có trong data → fallback về tuần gần nhất trong data.
    Luôn trả về zero-padded: "Tuần 08", "Tuần 20".
    """
    if not intent.has_temporal_reference:
        return None

    if intent.explicit_week in ("current", "next", "previous"):
        latest_db_week_num = None
        if available_weeks:
            nums = [extract_week_number(w) for w in available_weeks if extract_week_number(w)]
            if nums:
                latest_db_week_num = max(nums)

        if latest_db_week_num is None:
            logger.warning(
                "resolve_week_filter: available_weeks rỗng hoặc chưa load. "
                "Tuần 'current' không resolve được. Trả về None → dùng recency boost."
            )
            return None

        if intent.explicit_week == "current":
            target = latest_db_week_num
        elif intent.explicit_week == "previous":
            target = latest_db_week_num - 1
        else:
            target = latest_db_week_num + 1

        if available_weeks:
            nums = sorted([extract_week_number(w) for w in available_weeks if extract_week_number(w)])
            if nums:
                max_week = max(nums)
                if target in nums:
                    logger.debug(f"DB week {target} found in data → Tuần {target:02d}")
                    return f"Tuần {target:02d}"
                clamped = max(nums[0], min(target, max_week))
                logger.debug(f"DB week {target} not in data → clamp to Tuần {clamped:02d}")
                return f"Tuần {clamped:02d}"
        return f"Tuần {target:02d}"

    if intent.explicit_week:
        m = re.search(r"(\d+)", intent.explicit_week)
        if m:
            return f"Tuần {int(m.group(1)):02d}"
        return intent.explicit_week

    return None


def resolve_day_filter(intent: TemporalIntent) -> tuple[int | None, str | None]:
    """
    Trả về (ngày, thứ trong tuần) từ TemporalIntent.
    'ngày 15' → (15, None).
    'thứ hai' → (None, 'Thứ 2').
    Ưu tiên day_of_week vì data lịch công tác có field day_of_week, không có field day.
    """
    if not intent.has_temporal_reference:
        return None, None
    if intent.explicit_day_of_week is not None:
        return None, intent.explicit_day_of_week
    if intent.explicit_day is not None:
        return intent.explicit_day, None
    return None, None


def resolve_month_filter(
    intent: TemporalIntent,
    *,
    latest_week_month: int | None = None,
) -> int | None:
    """
    Trả về tháng (1-12) từ TemporalIntent.
    'tháng này' → trả về tháng của tuần mới nhất trong DB.
    'tháng 5' → trả về 5.
    """
    if not intent.has_temporal_reference:
        return None
    if intent.explicit_month is not None:
        return intent.explicit_month
    if intent.temporal_type == "current_month":
        return latest_week_month
    return None


def resolve_year_filter(
    intent: TemporalIntent,
    *,
    latest_week_year: int | None = None,
) -> int | None:
    """
    Trả về năm từ TemporalIntent.
    'năm nay' → trả về năm của tuần mới nhất trong DB.
    'năm 2025' → trả về 2025.
    """
    if not intent.has_temporal_reference:
        return None
    if intent.explicit_year is not None:
        return intent.explicit_year
    if intent.temporal_type == "current_year":
        return latest_week_year
    return None


def resolve_timestamp_filter(
    intent: TemporalIntent,
    *,
    latest_week_month: int | None = None,
    latest_week_year: int | None = None,
) -> tuple[int | None, int | None]:
    """
    Chuyển TemporalIntent thành khoảng timestamp (milliseconds) cho Qdrant Range filter.
    Trả về (timestamp_from_ms, timestamp_to_ms). Dùng None cho bound không xác định.
    """
    if not intent.has_temporal_reference:
        return None, None

    year = intent.explicit_year or latest_week_year or 2026

    if intent.temporal_type == "specific_week":
        if intent.explicit_week_num:
            base = datetime(2025, 8, 4)
            week_start = base + timedelta(weeks=intent.explicit_week_num - 1)
            week_end = week_start + timedelta(days=6, hours=23, minutes=59, seconds=59)
            return (
                int(week_start.timestamp() * 1000),
                int(week_end.timestamp() * 1000),
            )

    if intent.temporal_type == "current_week":
        return None, None

    if intent.temporal_type in ("date_full", "date_short", "specific_day"):
        if intent.explicit_day is not None:
            month = intent.explicit_day_month or latest_week_month or 5
            year_override = intent.explicit_day_year or latest_week_year or 2026
            day_start = datetime(year_override, month, intent.explicit_day)
            day_end = day_start + timedelta(days=1) - timedelta(seconds=1)
            return (
                int(day_start.timestamp() * 1000),
                int(day_end.timestamp() * 1000),
            )

    if intent.temporal_type == "month" and intent.explicit_month:
        month = intent.explicit_month
        year_override = intent.explicit_year or latest_week_year or 2026
        first = datetime(year_override, month, 1)
        last_day = calendar.monthrange(year_override, month)[1]
        last = datetime(year_override, month, last_day, 23, 59, 59)
        return (
            int(first.timestamp() * 1000),
            int(last.timestamp() * 1000),
        )

    if intent.temporal_type == "current_month":
        if latest_week_month and latest_week_year:
            year_override = latest_week_year
            month = latest_week_month
            first = datetime(year_override, month, 1)
            last_day = calendar.monthrange(year_override, month)[1]
            last = datetime(year_override, month, last_day, 23, 59, 59)
            return (
                int(first.timestamp() * 1000),
                int(last.timestamp() * 1000),
            )

    return None, None
