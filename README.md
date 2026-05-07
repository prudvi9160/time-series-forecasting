# Case Study Forecasting

An end-to-end time-series forecasting system that trains and compares multiple state-of-the-art models across 50 US states, automatically selects the best performer, and serves predictions via REST API and interactive web UI.

## Features

- **Multi-Model Training**: ARIMA/SARIMA, Facebook Prophet, XGBoost, and LSTM neural networks
- **50-State Coverage**: Individual models trained for each US state
- **Intelligent Model Selection**: Automatic best-model selection based on validation metrics
- **REST API**: FastAPI endpoints for predictions and model training
- **Interactive UI**: Streamlit dashboard for exploratory analysis and forecasting
- **Feature Engineering**: Lag features, rolling statistics, temporal features, and holiday indicators
- **Time-Series Validation**: Proper train/validation split preventing data leakage

## Quick Start

### Prerequisites
- Python 3.12+
- Excel file with state, date, and sales/target columns

### Installation

```bash
pip install -r requirements.txt
```

### Data Preparation

Place your Excel file at `data/raw.xlsx`. The system automatically detects common column names:
- **State**: `state`, `state_name`, `region`
- **Date**: `date`, `datetime`, `day`, `report_date`
- **Target**: `value`, `target`, `total`, `sales`, `count`, `cases`

Clean the data:
```bash
python scripts/clean_data.py
```

### Train Models

Train all models across all states:
```bash
python scripts/train_models.py
```

### Run API

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Example prediction:
```bash
curl http://127.0.0.1:8000/predict/texas?horizon=8
```

### Run Web UI

```bash
streamlit run streamlit_app.py
```

Access at `http://localhost:8501`

## API Endpoints

- `GET /` - System status
- `GET /predict/{state}` - Forecast for specific state (default 7 days)
- `GET /health` - Health check
- `POST /train` - Trigger model retraining

## Architecture

- **app/pipeline.py**: Data loading, cleaning, and validation split
- **app/features.py**: Feature engineering (lags, rolling stats, temporal, holidays)
- **app/training.py**: Multi-model training and selection
- **app/model.py**: Artifact loading and prediction serving
- **app/main.py**: FastAPI application

## Performance

All 50 states trained; ARIMA selected as best model for optimal performance on this dataset.

## License

Proprietary - Case Study Project
