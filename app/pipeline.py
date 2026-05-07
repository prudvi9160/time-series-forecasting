from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


DATE_COLUMNS = ["date", "datetime", "day", "report_date"]
STATE_COLUMNS = ["state", "state_name", "region"]
TARGET_COLUMNS = ["value", "target", "total", "count", "cases", "sales", "y"]
OPTIONAL_COLUMNS = ["category"]


@dataclass
class ForecastBundle:
    data: pd.DataFrame
    state_column: str
    date_column: str
    target_column: str


@dataclass
class StateSplit:
    train: pd.DataFrame
    validation: pd.DataFrame


def _first_existing(columns: Iterable[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _demo_data() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=240, freq="D")
    rows = []
    for state_name, start, step in [("Texas", 50, 0.4), ("Florida", 35, 0.25), ("California", 70, 0.3)]:
        values = [start + (index * step) + (3 if index % 7 == 0 else 0) for index, _ in enumerate(dates)]
        frame = pd.DataFrame({"state": state_name, "date": dates, "value": values})
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _parse_mixed_date(value: object) -> pd.Timestamp | None:
    if isinstance(value, pd.Timestamp):
        return value

    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    if "-" in text and "/" not in text:
        parsed = pd.to_datetime(text, errors="coerce", dayfirst=True)
        if not pd.isna(parsed):
            return parsed

    if "/" in text:
        parsed = pd.to_datetime(text, errors="coerce", dayfirst=False)
        if not pd.isna(parsed):
            return parsed

    parsed = pd.to_datetime(text, errors="coerce", dayfirst=False)
    if not pd.isna(parsed):
        return parsed

    return pd.to_datetime(text, errors="coerce", dayfirst=True)


def _parse_numeric_series(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(r"[^\d\.-]", "", regex=True)
    cleaned = cleaned.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return pd.to_numeric(cleaned, errors="coerce")


def load_raw_data(source_path: str | Path) -> pd.DataFrame:
    path = Path(source_path)
    if path.exists():
        return pd.read_excel(path)
    return _demo_data()


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.copy()
    renamed.columns = [str(column).strip().lower() for column in renamed.columns]
    return renamed


def clean_dates(frame: pd.DataFrame) -> ForecastBundle:
    normalized = normalize_columns(frame)
    state_column = _first_existing(normalized.columns, STATE_COLUMNS) or "state"
    date_column = _first_existing(normalized.columns, DATE_COLUMNS)
    target_column = _first_existing(normalized.columns, TARGET_COLUMNS) or "value"
    category_column = _first_existing(normalized.columns, OPTIONAL_COLUMNS)

    working = normalized.copy()
    if state_column not in working.columns:
        working[state_column] = "all"
    if date_column is None:
        raise ValueError("No date column found. Expected one of: date, datetime, day, report_date.")
    if target_column not in working.columns:
        raise ValueError("No target column found. Expected one of: value, target, total, count, cases, sales, y.")

    working[date_column] = working[date_column].apply(_parse_mixed_date)
    working = working.dropna(subset=[date_column])
    working[target_column] = _parse_numeric_series(working[target_column])
    if category_column is not None and category_column in working.columns:
        working[category_column] = working[category_column].astype(str).str.strip()

    completed_parts: list[pd.DataFrame] = []
    for state_name, group in working.groupby(state_column, dropna=False):
        ordered = group.sort_values(date_column).copy()
        date_range = pd.date_range(ordered[date_column].min(), ordered[date_column].max(), freq="D")
        completed = ordered.set_index(date_column).reindex(date_range)
        completed.index.name = date_column
        completed[state_column] = state_name
        completed[target_column] = completed[target_column].interpolate(limit_direction="both")
        completed[target_column] = completed[target_column].ffill().bfill()
        completed_parts.append(completed.reset_index())

    cleaned = pd.concat(completed_parts, ignore_index=True)
    cleaned = cleaned.rename(columns={date_column: "date", target_column: "value", state_column: "state"})
    cleaned["state"] = cleaned["state"].astype(str).str.strip()
    cleaned["date"] = cleaned["date"].apply(_parse_mixed_date)
    cleaned["value"] = _parse_numeric_series(cleaned["value"])
    cleaned = cleaned.dropna(subset=["value"])
    cleaned = cleaned.sort_values(["state", "date"]).reset_index(drop=True)
    return ForecastBundle(data=cleaned, state_column="state", date_column="date", target_column="value")


def save_cleaned_data(frame: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_excel(path, index=False)
    return path


def clean_excel_file(source_path: str | Path, output_path: str | Path) -> Path:
    raw = load_raw_data(source_path)
    bundle = clean_dates(raw)
    return save_cleaned_data(bundle.data, output_path)


def split_train_validation(state_frame: pd.DataFrame, validation_days: int = 56) -> StateSplit:
    ordered = state_frame.sort_values("date").reset_index(drop=True)
    if len(ordered) <= validation_days:
        raise ValueError("Not enough rows for a time-series validation split.")
    return StateSplit(train=ordered.iloc[:-validation_days].copy(), validation=ordered.iloc[-validation_days:].copy())
