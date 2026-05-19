"""
Data Loading Module - Trích xuất text từ các nguồn dữ liệu.
Hỗ trợ: JSONL (nguồn chính), PDF, Word, Excel, TXT.
"""
import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# Map thứ trong tuần
DOW_MAP: dict[str, str] = {
    # Full names (lowercase, for regex)
    "thứ hai": "Thứ 2", "thứ 2": "Thứ 2",
    "thứ ba": "Thứ 3", "thứ 3": "Thứ 3",
    "thứ tư": "Thứ 4", "thứ 4": "Thứ 4",
    "thứ năm": "Thứ 5", "thứ 5": "Thứ 5",
    "thứ sáu": "Thứ 6", "thứ 6": "Thứ 6",
    "thứ bảy": "Thứ 7", "thứ 7": "Thứ 7",
    "chủ nhật": "Chủ Nhật", "cn": "Chủ Nhật",
    # Capitalized full names (for direct lookup)
    "Thứ Hai": "Thứ 2", "Thứ 2": "Thứ 2",
    "Thứ Ba": "Thứ 3", "Thứ 3": "Thứ 3",
    "Thứ Tư": "Thứ 4", "Thứ 4": "Thứ 4",
    "Thứ Năm": "Thứ 5", "Thứ 5": "Thứ 5",
    "Thứ Sáu": "Thứ 6", "Thứ 6": "Thứ 6",
    "Thứ Bảy": "Thứ 7", "Thứ 7": "Thứ 7",
    "Chủ Nhật": "Chủ Nhật",
    # Abbreviations (uppercase in source: "Vào TƯ – 05/11")
    "tƯ": "Thứ 4", "tư": "Thứ 4",
    "sáU": "Thứ 6", "sáu": "Thứ 6",
    "nĂm": "Thứ 5", "năm": "Thứ 5",
    "bA": "Thứ 3", "ba": "Thứ 3",
    "hAi": "Thứ 2", "hai": "Thứ 2",
    "bảY": "Thứ 7", "bảy": "Thứ 7",
    # Standalone uppercase abbreviations
    "TƯ": "Thứ 4", "SÁU": "Thứ 6", "NĂM": "Thứ 5", "BA": "Thứ 3",
    "HAI": "Thứ 2", "BẢY": "Thứ 7",
    # "Thứ B" style (from Vietnamese week calendars)
    "thứ b": "Thứ 3", "thứ bảy": "Thứ 7",
    "Thứ B": "Thứ 3", "Thứ Bảy": "Thứ 7",
}


def extract_date_metadata(content: str) -> dict[str, Any]:
    """
    Trích xuất ngày/tháng/năm/thứ từ content dạng:
    "Vào Thứ Hai ngày 04 tháng 05 năm 2026..."
    Trả về dict: day, day_month, day_year, day_of_week (hoặc rỗng nếu không parse được).
    """
    result: dict[str, Any] = {}

    # Pattern: "Thứ Hai ngày 04 tháng 05 năm 2026" hoặc "ngày 04/05/2026"
    m = re.search(
        r"(?:Vào\s+)?(?:Thứ\s*[27三四五六báymội]\s*)?ngày\s+(\d{1,2})\s*(?:tháng\s+(\d{1,2}))?\s*(?:năm\s+(\d{4}))?",
        content, re.IGNORECASE,
    )
    if m:
        result["day"] = int(m.group(1))
        if m.group(2):
            result["day_month"] = int(m.group(2))
        if m.group(3):
            result["day_year"] = int(m.group(3))
        else:
            # infer year from content if available
            year_m = re.search(r"năm\s+(\d{4})", content, re.IGNORECASE)
            if year_m:
                result["day_year"] = int(year_m.group(1))

    # Extract day_of_week: "Thứ Hai", "Thứ 3", "TƯ", "SÁU", "BA", "Chủ Nhật"
    # Priority: find first match among multiple patterns
    if "day_of_week" not in result:
        dow_m = re.search(
            r"(thứ\s*[27三四五六báymội]|chủ\s*nhật|cn)",
            content, re.IGNORECASE,
        )
        if dow_m:
            raw = dow_m.group(0).lower()
            result["day_of_week"] = DOW_MAP.get(raw, dow_m.group(0).title())

    # Standalone abbreviations: "Vào TƯ – 05/11", "Vào SÁU – 10/10", etc.
    if "day_of_week" not in result:
        abbrev_m = re.search(
            r"(?:Vào\s+)?([A-ZÁÀẢẠÃĂẰẮẲẶÂẦấ][\wÀ-ỹ]*)\s*[–\-]\s*\d",
            content,
        )
        if abbrev_m:
            raw = abbrev_m.group(1)
            result["day_of_week"] = DOW_MAP.get(raw, DOW_MAP.get(raw.upper(), DOW_MAP.get(raw.lower())))

    # "Thứ B" style (2-char abbrev in some calendars)
    if "day_of_week" not in result:
        thu_b_m = re.search(r"Thứ\s*([BM2-7])", content, re.IGNORECASE)
        if thu_b_m:
            abbrev_map = {"B": "Thứ 3", "M": "Thứ 2", "2": "Thứ 2", "3": "Thứ 3",
                          "4": "Thứ 4", "5": "Thứ 5", "6": "Thứ 6", "7": "Thứ 7"}
            result["day_of_week"] = abbrev_map.get(thu_b_m.group(1).upper(), thu_b_m.group(0))

    # Fallback: parse "ngày dd/mm" or "dd/mm/yyyy"
    if "day" not in result:
        slash_m = re.search(r"(\d{1,2})/(\d{1,2})(?:/(\d{4}))?", content)
        if slash_m:
            result["day"] = int(slash_m.group(1))
            result["day_month"] = int(slash_m.group(2))
            if slash_m.group(3):
                result["day_year"] = int(slash_m.group(3))

    return result


def load_jsonl(file_path: str) -> list[dict[str, Any]]:
    """Đọc file JSONL, trả về list các bản ghi. Tự động enrich metadata với date fields."""
    records = []
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Dòng {line_num} bị lỗi: {e}")
                continue

            # Enrich metadata with date fields extracted from content
            content = record.get("content", "")
            if content:
                date_meta = extract_date_metadata(content)
                if "metadata" not in record:
                    record["metadata"] = {}
                record["metadata"].update(date_meta)

            records.append(record)

    logger.info(f"Đã đọc {len(records)} bản ghi từ {file_path}")
    return records


def load_jsonl_as_dataframe(file_path: str) -> pd.DataFrame:
    """Đọc JSONL và trả về DataFrame để phân tích."""
    records = load_jsonl(file_path)
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    if "metadata" in df.columns:
        meta_df = df["metadata"].apply(pd.Series)
        df = pd.concat([df.drop(columns=["metadata"]), meta_df], axis=1)
    return df


def load_raw_files(raw_dir: str) -> list[dict[str, Any]]:
    """Duyệt thư mục raw, trích xuất text từ mọi định dạng được hỗ trợ."""
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        logger.warning(f"Thư mục raw không tồn tại: {raw_dir}")
        return []

    all_records = []
    for file_path in raw_path.rglob("*"):
        if file_path.is_file():
            ext = file_path.suffix.lower()
            try:
                if ext == ".jsonl":
                    records = load_jsonl(str(file_path))
                    all_records.extend(records)
                elif ext == ".txt":
                    records.append(_load_txt(file_path))
                elif ext in (".xlsx", ".xls"):
                    records = _load_excel(file_path)
                    all_records.extend(records)
                elif ext == ".pdf":
                    logger.info(f"PDF chưa được cài đặt extractor riêng: {file_path}")
                elif ext in (".docx", ".doc"):
                    logger.info(f"Word chưa được cài đặt extractor riêng: {file_path}")
            except Exception as e:
                logger.error(f"Lỗi đọc {file_path}: {e}")

    return all_records


def _load_txt(file_path: Path) -> dict[str, Any]:
    """Trích xuất text từ file TXT."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return {
        "id": file_path.stem,
        "content": content,
        "metadata": {"source": str(file_path), "format": "txt"},
    }


def _load_excel(file_path: Path) -> list[dict[str, Any]]:
    """Trích xuất text từ file Excel, mỗi dòng thành một bản ghi."""
    records = []
    df = pd.read_excel(file_path, engine="openpyxl")
    for idx, row in df.iterrows():
        text_parts = [f"{col}: {val}" for col, val in row.items() if pd.notna(val)]
        if text_parts:
            records.append({
                "id": f"{file_path.stem}_row_{idx}",
                "content": " | ".join(text_parts),
                "metadata": {"source": str(file_path), "format": "excel", "row": idx},
            })
    logger.info(f"Đã đọc {len(records)} dòng từ {file_path}")
    return records
