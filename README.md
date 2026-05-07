## SALES FORECASTING DOCUMENTATION

This is a draft version of the project focused on one working path first:

1. Clean an Excel file and fill missing dates.
2. Engineer time-series features with no leakage.
3. Train ARIMA, Prophet, XGBoost, and LSTM per state.
4. Compare validation performance and select the best model.
5. Expose predictions through a REST API.

## Expected input

Place your Excel file at `data/raw.xlsx` or set `DATA_SOURCE_PATH` to the file you want to use.

The cleaner looks for common column names:

- state: `state`, `state_name`, `region`
- date: `date`, `datetime`, `day`, `report_date`
- target: `value`, `target`, `total`, `count`, `cases`, `sales`, `y`

Your sheet schema is supported directly:

- `State` -> state
- `Date` -> date
- `Total` -> target
- `Category` -> optional context field

## Run the cleaner

```bash
python scripts/clean_data.py
```

This writes a cleaned Excel file to `data/cleaned.xlsx` by default.

## Train all models

```bash
python scripts/train_models.py
```

This creates per-state artifacts in `artifacts/` with the best model and validation metrics.

If you start the API first, you can also trigger the same process with `POST /train`.

## Run the API

```bash
uvicorn app.main:app --reload
```

Example request:

```bash
curl http://127.0.0.1:8000/predict/texas?horizon=56
```

## Draft scope

The other model ideas and extra feature engineering can be expanded later, but the current code already includes the mandatory four model families and time-series validation.
