from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from app.features import FeatureConfig, make_feature_matrix
from app.pipeline import ForecastBundle, split_train_validation


@dataclass
class ModelMetrics:
    model_name: str
    mae: float
    rmse: float
    mape: float


@dataclass
class StateModelArtifact:
    state: str
    best_model: str
    metrics: list[ModelMetrics]
    artifact_path: str
    validation_days: int


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.clip(np.abs(y_true), 1e-6, None)
    return float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100)


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> ModelMetrics:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return ModelMetrics(
        model_name="",
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=rmse,
        mape=_mape(y_true, y_pred),
    )


def _predict_arima(train: pd.Series, validation_length: int) -> np.ndarray:
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except Exception:
        return np.repeat(float(train.iloc[-1]), validation_length)

    try:
        order_candidates = [(1, 1, 1), (2, 1, 1)]
        best_forecast = None
        best_aic = np.inf
        for order in order_candidates:
            model = SARIMAX(train, order=order, seasonal_order=(1, 1, 1, 7), enforce_stationarity=False, enforce_invertibility=False)
            fitted = model.fit(disp=False)
            forecast = fitted.forecast(validation_length)
            if fitted.aic < best_aic:
                best_aic = fitted.aic
                best_forecast = forecast
        return np.asarray(best_forecast)
    except Exception:
        return np.repeat(float(train.iloc[-1]), validation_length)


def _predict_prophet(train_frame: pd.DataFrame, validation_dates: pd.Series) -> np.ndarray:
    try:
        from prophet import Prophet
    except Exception:
        return np.repeat(float(train_frame["value"].iloc[-1]), len(validation_dates))

    try:
        model_frame = train_frame[["date", "value"]].rename(columns={"date": "ds", "value": "y"})
        model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
        model.fit(model_frame)
        future = pd.DataFrame({"ds": pd.to_datetime(validation_dates)})
        forecast = model.predict(future)
        return forecast["yhat"].to_numpy()
    except Exception:
        return np.repeat(float(train_frame["value"].iloc[-1]), len(validation_dates))


def _predict_xgboost(train_frame: pd.DataFrame, validation_frame: pd.DataFrame, config: FeatureConfig) -> np.ndarray:
    try:
        import xgboost as xgb
    except Exception:
        return np.repeat(float(train_frame["value"].iloc[-1]), len(validation_frame))

    train_features = make_feature_matrix(train_frame, target_column="value", config=config)
    validation_features = make_feature_matrix(pd.concat([train_frame.tail(40), validation_frame], ignore_index=True), target_column="value", config=config)
    validation_features = validation_features[validation_features["date"].isin(validation_frame["date"])]

    feature_columns = [column for column in train_features.columns if column not in {"date", "state", "value"}]
    X_train = train_features[feature_columns]
    y_train = train_features["value"]
    X_validation = validation_features[feature_columns]

    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model.predict(X_validation)


def _build_lstm_sequences(frame: pd.DataFrame, lookback: int = 30) -> tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    values = frame[["value"]].astype(float).to_numpy()
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(values)

    sequences: list[np.ndarray] = []
    targets: list[float] = []
    for index in range(lookback, len(scaled_values)):
        sequences.append(scaled_values[index - lookback:index])
        targets.append(scaled_values[index, 0])
    return np.asarray(sequences), np.asarray(targets), scaler


def _predict_lstm(train_frame: pd.DataFrame, validation_frame: pd.DataFrame) -> np.ndarray:
    try:
        import tensorflow as tf
    except Exception:
        return np.repeat(float(train_frame["value"].iloc[-1]), len(validation_frame))

    try:
        lookback = 30
        combined = pd.concat([train_frame, validation_frame], ignore_index=True).sort_values("date").reset_index(drop=True)
        sequences, targets, scaler = _build_lstm_sequences(train_frame, lookback=lookback)
        if len(sequences) < 10:
            return np.repeat(float(train_frame["value"].iloc[-1]), len(validation_frame))

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(lookback, 1)),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(sequences, targets, epochs=15, batch_size=16, verbose=0)

        history = train_frame["value"].astype(float).tolist()
        predictions: list[float] = []
        for _ in range(len(validation_frame)):
            recent = np.asarray(history[-lookback:], dtype=float).reshape(-1, 1)
            recent_scaled = scaler.transform(recent).reshape(1, lookback, 1)
            next_scaled = model.predict(recent_scaled, verbose=0)[0, 0]
            next_value = float(scaler.inverse_transform([[next_scaled]])[0, 0])
            predictions.append(next_value)
            history.append(next_value)
        return np.asarray(predictions)
    except Exception:
        return np.repeat(float(train_frame["value"].iloc[-1]), len(validation_frame))


def train_state_models(bundle: ForecastBundle, state: str, validation_days: int = 56, artifacts_dir: str | Path = "artifacts") -> StateModelArtifact:
    state_frame = bundle.data[bundle.data[bundle.state_column].str.lower() == state.strip().lower()].copy()
    if state_frame.empty:
        available_states = sorted(bundle.data[bundle.state_column].dropna().astype(str).unique().tolist())
        raise ValueError(f"Unknown state '{state}'. Available states: {', '.join(available_states)}")

    split = split_train_validation(state_frame, validation_days=validation_days)
    config = FeatureConfig()
    train_series = split.train["value"].astype(float)
    validation_series = split.validation["value"].astype(float).to_numpy()

    metrics: list[ModelMetrics] = []
    predictions: dict[str, np.ndarray] = {}

    predictions["arima"] = _predict_arima(train_series, len(split.validation))
    predictions["prophet"] = _predict_prophet(split.train, split.validation["date"])
    predictions["xgboost"] = _predict_xgboost(split.train, split.validation, config)
    predictions["lstm"] = _predict_lstm(split.train, split.validation)

    for model_name, forecast in predictions.items():
        model_metrics = _evaluate(validation_series, np.asarray(forecast, dtype=float))
        model_metrics.model_name = model_name
        metrics.append(model_metrics)

    best_metric = min(metrics, key=lambda metric: (metric.mae, metric.rmse))
    state_dir = Path(artifacts_dir) / state.strip().lower()
    state_dir.mkdir(parents=True, exist_ok=True)

    artifact = {
        "state": state,
        "best_model": best_metric.model_name,
        "metrics": [asdict(metric) for metric in metrics],
        "validation_days": validation_days,
        "history": split.train.to_dict(orient="records"),
        "target_column": bundle.target_column,
        "state_column": bundle.state_column,
        "date_column": bundle.date_column,
    }
    artifact_path = state_dir / "artifact.joblib"
    joblib.dump(artifact, artifact_path)

    with open(state_dir / "metrics.json", "w", encoding="utf-8") as file_handle:
        json.dump({"state": state, "best_model": best_metric.model_name, "metrics": [asdict(metric) for metric in metrics]}, file_handle, indent=2)

    return StateModelArtifact(
        state=state,
        best_model=best_metric.model_name,
        metrics=metrics,
        artifact_path=str(artifact_path),
        validation_days=validation_days,
    )


def train_all_states(bundle: ForecastBundle, validation_days: int = 56, artifacts_dir: str | Path = "artifacts") -> list[StateModelArtifact]:
    artifacts: list[StateModelArtifact] = []
    for state in sorted(bundle.data[bundle.state_column].dropna().astype(str).unique().tolist()):
        artifacts.append(train_state_models(bundle, state=state, validation_days=validation_days, artifacts_dir=artifacts_dir))
    return artifacts
