from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import holidays
import pandas as pd


@dataclass
class FeatureConfig:
    holiday_country: str = "US"
    lags: tuple[int, ...] = (1, 7, 30)
    rolling_windows: tuple[int, ...] = (7, 30)


def add_time_features(frame: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    featured = frame.copy()
    featured[date_column] = pd.to_datetime(featured[date_column])
    featured["day_of_week"] = featured[date_column].dt.dayofweek
    featured["month"] = featured[date_column].dt.month
    featured["day_of_month"] = featured[date_column].dt.day
    featured["week_of_year"] = featured[date_column].dt.isocalendar().week.astype(int)
    return featured


def add_holiday_flag(frame: pd.DataFrame, date_column: str = "date", country: str = "US") -> pd.DataFrame:
    featured = frame.copy()
    calendar = holidays.country_holidays(country, years=sorted(featured[date_column].dt.year.unique().tolist()))
    featured["holiday_flag"] = featured[date_column].dt.date.apply(lambda current_date: int(current_date in calendar))
    return featured


def add_lag_and_rolling_features(frame: pd.DataFrame, target_column: str = "value", config: FeatureConfig | None = None) -> pd.DataFrame:
    config = config or FeatureConfig()
    featured = frame.copy().sort_values("date").reset_index(drop=True)

    for lag in config.lags:
        featured[f"lag_{lag}"] = featured[target_column].shift(lag)

    for window in config.rolling_windows:
        shifted = featured[target_column].shift(1)
        featured[f"rolling_mean_{window}"] = shifted.rolling(window=window).mean()
        featured[f"rolling_std_{window}"] = shifted.rolling(window=window).std()

    featured = add_time_features(featured, date_column="date")
    featured = add_holiday_flag(featured, date_column="date", country=config.holiday_country)
    return featured


def make_feature_matrix(frame: pd.DataFrame, target_column: str = "value", config: FeatureConfig | None = None) -> pd.DataFrame:
    featured = add_lag_and_rolling_features(frame, target_column=target_column, config=config)
    featured = featured.dropna().reset_index(drop=True)
    return featured
