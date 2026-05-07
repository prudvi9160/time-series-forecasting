from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException

from app.model import forecast_state
from app.pipeline import clean_dates, load_raw_data
from app.training import train_all_states


app = FastAPI(title="QuickHyreAI Draft API", version="0.1.0")


@lru_cache(maxsize=1)
def get_bundle():
    source_path = os.getenv("DATA_SOURCE_PATH", "data/raw.xlsx")
    raw = load_raw_data(source_path)
    return clean_dates(raw)


def list_trained_states(artifacts_dir: str) -> list[str]:
    base_path = Path(artifacts_dir)
    if not base_path.exists():
        return []
    return sorted([path.name for path in base_path.iterdir() if path.is_dir() and (path / "artifact.joblib").exists()])


@app.get("/")
def root():
    bundle = get_bundle()
    artifacts_dir = os.getenv("ARTIFACTS_DIR", "artifacts")
    return {
        "status": "ok",
        "message": "Draft API is running",
        "available_states": sorted(bundle.data[bundle.state_column].dropna().astype(str).unique().tolist()),
        "trained_states": list_trained_states(artifacts_dir),
    }


@app.post("/train")
def train_models():
    bundle = get_bundle()
    validation_days = int(os.getenv("VALIDATION_DAYS", "56"))
    artifacts_dir = os.getenv("ARTIFACTS_DIR", "artifacts")
    artifacts = train_all_states(bundle, validation_days=validation_days, artifacts_dir=artifacts_dir)
    return {
        "status": "trained",
        "artifacts": [
            {
                "state": artifact.state,
                "best_model": artifact.best_model,
                "artifact_path": artifact.artifact_path,
                "validation_days": artifact.validation_days,
            }
            for artifact in artifacts
        ],
    }


@app.get("/predict/{state}")
def predict(state: str, horizon: int = 7):
    try:
        bundle = get_bundle()
        artifacts_dir = os.getenv("ARTIFACTS_DIR", "artifacts")
        result = forecast_state(bundle, state=state, horizon=horizon, artifacts_dir=artifacts_dir)
        return {
            "state": result.state,
            "horizon": result.horizon,
            "model_type": result.model_type,
            "forecast": result.forecast,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/health")
def health():
    return {"status": "healthy"}
