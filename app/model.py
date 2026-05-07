from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.features import FeatureConfig, make_feature_matrix
from app.pipeline import ForecastBundle


@dataclass
class ForecastResult:
    state: str
    horizon: int
    forecast: list[dict[str, Any]]
    model_type: str


def _baseline_forecast(history_frame: pd.DataFrame, horizon: int) -> list[dict[str, Any]]:
    ordered = history_frame.sort_values("date").copy()
    last_date = ordered["date"].max()
    last_value = float(ordered["value"].iloc[-1])
    recent = ordered["value"].tail(min(7, len(ordered)))
    daily_step = float(recent.diff().dropna().mean()) if len(recent) > 1 else 0.0
    if pd.isna(daily_step):
        daily_step = 0.0

    rows: list[dict[str, Any]] = []
    for step in range(1, horizon + 1):
        rows.append(
            {
                "date": (last_date + pd.Timedelta(days=step)).date().isoformat(),
                "prediction": round(last_value + (daily_step * step), 2),
                "source": "baseline",
            }
        )
    return rows


def _fit_prophet(train_frame: pd.DataFrame, horizon_dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        from prophet import Prophet
    except Exception:
        predictions = np.repeat(float(train_frame["value"].iloc[-1]), len(horizon_dates))
        return predictions, predictions, predictions

    model_frame = train_frame[["date", "value"]].rename(columns={"date": "ds", "value": "y"})
    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    model.fit(model_frame)
    forecast = model.predict(pd.DataFrame({"ds": horizon_dates}))
    return forecast["yhat"].to_numpy(), forecast["yhat_lower"].to_numpy(), forecast["yhat_upper"].to_numpy()


def _fit_arima(train_frame: pd.DataFrame, horizon: int) -> list[dict[str, Any]]:
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except Exception:
        return _baseline_forecast(train_frame, horizon)

    series = train_frame.sort_values("date")["value"].astype(float)
    try:
        model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7), enforce_stationarity=False, enforce_invertibility=False)
        fitted = model.fit(disp=False)
        forecast = fitted.get_forecast(steps=horizon)
        frame = forecast.summary_frame(alpha=0.05)
        last_date = train_frame["date"].max()
        rows: list[dict[str, Any]] = []
        for offset, (_, row) in enumerate(frame.iterrows(), start=1):
            rows.append(
                {
                    "date": (last_date + pd.Timedelta(days=offset)).date().isoformat(),
                    "prediction": round(float(row["mean"]), 2),
                    "lower": round(float(row["mean_ci_lower"]), 2),
                    "upper": round(float(row["mean_ci_upper"]), 2),
                    "source": "arima",
                }
            )
        return rows
    except Exception:
        return _baseline_forecast(train_frame, horizon)


def _recursive_xgboost_forecast(train_frame: pd.DataFrame, horizon: int, config: FeatureConfig) -> list[float]:
    try:
        import xgboost as xgb
    except Exception:
        return [float(train_frame["value"].iloc[-1])] * horizon

    feature_frame = make_feature_matrix(train_frame, target_column="value", config=config)
    if feature_frame.empty:
        return [float(train_frame["value"].iloc[-1])] * horizon

    feature_columns = [column for column in feature_frame.columns if column not in {"date", "state", "value"}]
    model = xgb.XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(feature_frame[feature_columns], feature_frame["value"])

    history = train_frame.sort_values("date").copy()
    predictions: list[float] = []
    for step in range(horizon):
        next_date = history["date"].max() + pd.Timedelta(days=1)
        next_row = _build_next_feature_row(history, next_date, config)
        prediction = float(model.predict(next_row[feature_columns])[0])
        predictions.append(prediction)
        history = pd.concat([history, pd.DataFrame({"date": [next_date], "state": [history["state"].iloc[-1]], "value": [prediction]})], ignore_index=True)
    return predictions


def _build_next_feature_row(history_frame: pd.DataFrame, next_date: pd.Timestamp, config: FeatureConfig) -> pd.DataFrame:
    ordered = history_frame.sort_values("date").reset_index(drop=True)
    history_values = ordered["value"].astype(float)
    row: dict[str, Any] = {
        "date": next_date,
        "state": ordered["state"].iloc[-1],
        "value": float(history_values.iloc[-1]),
        "day_of_week": next_date.dayofweek,
        "month": next_date.month,
        "day_of_month": next_date.day,
        "week_of_year": int(next_date.isocalendar().week),
        "holiday_flag": 0,
    }
    try:
        import holidays

        calendar = holidays.country_holidays(config.holiday_country, years=[next_date.year])
        row["holiday_flag"] = int(next_date.date() in calendar)
    except Exception:
        row["holiday_flag"] = 0

    for lag in config.lags:
        row[f"lag_{lag}"] = float(history_values.iloc[-lag]) if len(history_values) >= lag else float(history_values.iloc[0])

    for window in config.rolling_windows:
        source = history_values.shift(1).tail(window)
        row[f"rolling_mean_{window}"] = float(source.mean()) if not source.empty else float(history_values.mean())
        row[f"rolling_std_{window}"] = float(source.std()) if len(source) > 1 else 0.0

    return pd.DataFrame([row])


def _recursive_lstm_forecast(train_frame: pd.DataFrame, horizon: int) -> list[float]:
    try:
        import tensorflow as tf
        from sklearn.preprocessing import MinMaxScaler
    except Exception:
        return [float(train_frame["value"].iloc[-1])] * horizon

    lookback = 30
    values = train_frame[["value"]].astype(float).to_numpy()
    if len(values) <= lookback:
        return [float(train_frame["value"].iloc[-1])] * horizon

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)
    sequences: list[np.ndarray] = []
    targets: list[float] = []
    for index in range(lookback, len(scaled)):
        sequences.append(scaled[index - lookback:index])
        targets.append(scaled[index, 0])

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(lookback, 1)),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.asarray(sequences), np.asarray(targets), epochs=10, batch_size=16, verbose=0)

    history = values.flatten().tolist()
    predictions: list[float] = []
    for _ in range(horizon):
        recent = np.asarray(history[-lookback:], dtype=float).reshape(-1, 1)
        recent_scaled = scaler.transform(recent).reshape(1, lookback, 1)
        next_scaled = float(model.predict(recent_scaled, verbose=0)[0, 0])
        next_value = float(scaler.inverse_transform([[next_scaled]])[0, 0])
        predictions.append(next_value)
        history.append(next_value)
    return predictions


def _load_artifact(state: str, artifacts_dir: str | Path = "artifacts") -> dict[str, Any]:
    artifact_path = Path(artifacts_dir) / state.strip().lower() / "artifact.joblib"
    if not artifact_path.exists():
        raise FileNotFoundError(f"No trained artifact found for state '{state}'. Run scripts/train_models.py first.")
    return joblib.load(artifact_path)


def forecast_state(bundle: ForecastBundle, state: str, horizon: int = 56, artifacts_dir: str | Path = "artifacts") -> ForecastResult:
    state_key = state.strip().lower()
    matched = bundle.data[bundle.data[bundle.state_column].str.lower() == state_key].copy()
    if matched.empty:
        available_states = sorted(bundle.data[bundle.state_column].dropna().astype(str).unique().tolist())
        raise ValueError(f"Unknown state '{state}'. Available states: {', '.join(available_states)}")

    try:
        artifact = _load_artifact(state, artifacts_dir=artifacts_dir)
        best_model = artifact["best_model"]
        history = pd.DataFrame(artifact["history"])
        history["date"] = pd.to_datetime(history["date"])
    except FileNotFoundError:
        history = matched.sort_values("date").copy()
        best_model = "baseline"

    forecast_dates = pd.date_range(history["date"].max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    config = FeatureConfig()

    if best_model == "prophet":
        yhat, lower, upper = _fit_prophet(history, forecast_dates)
        forecast_rows = [
            {
                "date": date_value.date().isoformat(),
                "prediction": round(float(prediction), 2),
                "lower": round(float(lower_value), 2),
                "upper": round(float(upper_value), 2),
                "source": "prophet",
            }
            for date_value, prediction, lower_value, upper_value in zip(forecast_dates, yhat, lower, upper)
        ]
    elif best_model == "arima":
        forecast_rows = _fit_arima(history, horizon)
    elif best_model == "xgboost":
        predictions = _recursive_xgboost_forecast(history, horizon, config)
        forecast_rows = [
            {"date": date_value.date().isoformat(), "prediction": round(float(prediction), 2), "source": "xgboost"}
            for date_value, prediction in zip(forecast_dates, predictions)
        ]
    elif best_model == "lstm":
        predictions = _recursive_lstm_forecast(history, horizon)
        forecast_rows = [
            {"date": date_value.date().isoformat(), "prediction": round(float(prediction), 2), "source": "lstm"}
            for date_value, prediction in zip(forecast_dates, predictions)
        ]
    else:
        forecast_rows = _baseline_forecast(history, horizon)

    return ForecastResult(state=state, horizon=horizon, forecast=forecast_rows, model_type=best_model)
