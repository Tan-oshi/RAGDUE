"""
QueryParser Schemas — Pydantic models cho structured query parsing.
Dùng chung cho QueryParser và benchmark shim.
"""
from enum import Enum

from pydantic import BaseModel, Field


class QueryType(str, Enum):
    """Các loại câu hỏi được phân loại bởi QueryParser."""

    LIST = "list"          # Liệt kê sự kiện
    COUNT = "count"        # Đếm số lượng
    WHO = "who"            # Hỏi người tham gia / chủ trì
    WHEN = "when"          # Hỏi thời gian
    WHERE = "where"        # Hỏi địa điểm
    DETAIL = "detail"      # Hỏi chi tiết
    GENERAL = "general"    # Câu hỏi chung


class TemporalSpec(BaseModel):
    """
    Thông tin thời gian được trích xuất từ câu hỏi.
    Thay thế hoàn toàn regex trong temporal.py.
    """

    has_reference: bool = Field(
        default=False,
        description="Câu hỏi có chứa tham chiếu thời gian hay không"
    )
    type: str = Field(
        default="general",
        description="Loại temporal: general, year, month, week, day_of_week, date, semester, next_week, previous_week, current_week"
    )
    year: int | None = Field(default=None, description="Năm cụ thể (VD: 2025)")
    month: int | None = Field(default=None, description="Tháng cụ thể (1-12)")
    day: int | None = Field(default=None, description="Ngày cụ thể (1-31)")
    day_of_week: str | None = Field(
        default=None,
        description="Thứ trong tuần: 'Thứ Hai', 'Thứ Ba', ..., 'Chủ Nhật'"
    )
    week: str | None = Field(
        default=None,
        description="Tuần cụ thể (VD: 'Tuần 40', 'Tuần 15')"
    )
    relative: str | None = Field(
        default=None,
        description="Relative temporal: next_week, previous_week, current_week"
    )

    def resolve_to_week_filter(self, available_weeks: list[str]) -> str | None:
        """
        Resolve temporal spec to a week filter string.
        Implement logic equivalent to resolve_week_filter() from temporal.py.
        """
        if self.relative == "next_week":
            return "next"
        if self.relative == "previous_week":
            return "previous"
        if self.week:
            return self.week
        if self.year and self.month:
            for w in available_weeks:
                if str(self.year) in w and f"{self.month:02d}" in w:
                    return w
        return None


class ContentSpec(BaseModel):
    """
    Thông tin nội dung được trích xuất từ câu hỏi.
    Thay thế hoàn toàn regex trong content_filter.py + query_intent.py.
    """

    is_list_query: bool = Field(
        default=False,
        description="Câu hỏi yêu cầu liệt kê (thay vì count/who)"
    )
    chairperson: str | None = Field(
        default=None,
        description="Tên người chủ trì (VD: 'Hiệu trưởng', 'Phó Hiệu trưởng')"
    )
    event_name: str | None = Field(
        default=None,
        description="Tên sự kiện/cuộc họp (VD: 'giao ban', 'tuyển dụng')"
    )
    participants: str | None = Field(
        default=None,
        description="Người tham gia được đề cập"
    )
    location: str | None = Field(
        default=None,
        description="Địa điểm được đề cập"
    )


class ParsedQuery(BaseModel):
    """
    Kết quả đầy đủ của QueryParser.parse().
    Dùng bởi cả benchmark shim và production QueryParser.
    """

    temporal: TemporalSpec = Field(default_factory=TemporalSpec)
    content: ContentSpec = Field(default_factory=ContentSpec)
    query_type: QueryType = Field(default=QueryType.LIST)
    is_general_query: bool = Field(
        default=False,
        description="True = câu hỏi không có temporal reference (is_general_query)"
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Độ tin cậy của việc parse (0.0-1.0)"
    )
