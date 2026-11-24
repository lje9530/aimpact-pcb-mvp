from __future__ import annotations

import csv
import re
from datetime import datetime
from io import BytesIO, StringIO
from pathlib import Path
from zipfile import BadZipFile

from openpyxl import load_workbook
from openpyxl.utils.exceptions import InvalidFileException


FIELDS = {
    "external_id": ("검사 ID", "inspection_id", "id", "관리번호"),
    "product": ("제품코드", "product_code", "product", "품번", "part_no"),
    "inspected_at": ("검사일시", "inspected_at", "날짜", "date", "timestamp"),
    "process": ("검사공정", "process", "공정", "station"),
    "batch": ("로트", "batch", "batch_id", "lot_no"),
    "reported_result": ("협력사 판정", "result", "판정", "판정결과", "judgement"),
    "reported_type": ("협력사 불량유형", "defect_type", "불량유형", "defect"),
    "image_file": ("검사 영상 파일명", "image_file", "image", "검사이미지", "aoi_file"),
    "reference_file": ("기준 영상 파일명", "reference_file", "reference", "기준이미지", "golden_file"),
}
REQUIRED = ("product", "inspected_at", "image_file")


def read_table(content, filename):
    ext = Path(filename).suffix.lower()
    if ext == ".csv":
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = content.decode("cp949")
        rows = csv.reader(StringIO(text))
        return table_rows(rows)
    if ext == ".xlsx":
        try:
            book = load_workbook(BytesIO(content), read_only=True, data_only=True)
        except (BadZipFile, InvalidFileException, KeyError) as exc:
            raise ValueError("엑셀 파일을 읽을 수 없습니다.") from exc
        try:
            sheet = book.active
            if (sheet.max_column or 0) > 80:
                raise ValueError("워크시트는 80열 이하여야 합니다.")
            return table_rows(sheet.values)
        finally:
            book.close()
    raise ValueError("CSV 또는 XLSX 파일을 선택하세요.")


def table_rows(rows):
    rows = iter(rows)
    headers = [str(value or "").strip() for value in next(rows, [])]
    if not headers or len(headers) > 80 or not all(headers) or len(set(headers)) != len(headers):
        raise ValueError("첫 행에는 비어 있거나 중복된 열 이름 없이 최대 80개 열을 입력하세요.")
    records = []
    for index, row in enumerate(rows, start=2):
        if not any(value is not None and str(value).strip() for value in row):
            continue
        if len(row) > len(headers) or len(records) >= 500:
            raise ValueError(f"{index}행의 열 수를 확인하세요. 한 번에 최대 500행까지 불러올 수 있습니다.")
        values = ["" if value is None else str(value).strip() for value in row]
        records.append((index, dict(zip(headers, values))))
    if not records:
        raise ValueError("파일에 검사 기록이 없습니다.")
    return headers, records


def suggest(headers):
    def key(value):
        return re.sub(r"[\s_\-]", "", value).casefold()

    lookup = {key(header): header for header in headers}
    return {field: next((lookup[key(alias)] for alias in (field, *aliases) if key(alias) in lookup), "")
            for field, aliases in FIELDS.items()}


def normalized(row, mapping):
    record = {field: row.get(mapping.get(field), "") for field in FIELDS}
    for field in REQUIRED:
        if not record[field]:
            raise ValueError(f"{FIELDS[field][0]} 값이 필요합니다.")
    if any(len(value) > 250 for value in record.values()):
        raise ValueError("검사 데이터 항목은 250자까지 입력할 수 있습니다.")
    try:
        value = record["inspected_at"].replace("/", "-")
        if re.fullmatch(r"\d{8}", value):
            value = datetime.strptime(value, "%Y%m%d").isoformat()
        record["inspected_at"] = datetime.fromisoformat(value).isoformat(timespec="seconds")
    except ValueError as exc:
        raise ValueError("검사일시는 YYYY-MM-DD 또는 YYYY-MM-DD HH:MM:SS 형식으로 입력하세요.") from exc
    record["process"] = record["process"] or "AOI 최종 검사"
    return record
