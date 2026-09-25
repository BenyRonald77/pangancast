"""Registry model forecasting."""

from __future__ import annotations

from src.models.base import Forecaster
from src.models.naive import NaiveForecaster, SeasonalNaiveForecaster

MODEL_LABELS = {
    "naive": "Naive",
    "seasonal_naive": "Seasonal Naive",
    "arima": "SARIMA",
    "prophet": "Prophet",
    "xgboost": "XGBoost",
    "lstm": "LSTM",
}
BASELINES = ("naive", "seasonal_naive")


def get_model(name: str, params: dict | None = None, seed: int = 42) -> Forecaster:
    """Buat instance model berdasarkan nama (impor lazy agar dashboard tetap ringan)."""
    params = dict(params or {})
    if name == "naive":
        return NaiveForecaster()
    if name == "seasonal_naive":
        return SeasonalNaiveForecaster(m=params.get("m", 7))
    if name == "arima":
        from src.models.arima import ArimaForecaster

        return ArimaForecaster(**params)
    if name == "prophet":
        from src.models.prophet_model import ProphetForecaster

        return ProphetForecaster(seed=seed, **params)
    if name == "xgboost":
        from src.models.xgb import XGBoostForecaster

        return XGBoostForecaster(seed=seed, **params)
    if name == "lstm":
        from src.models.lstm import LSTMForecaster

        return LSTMForecaster(seed=seed, **params)
    raise ValueError(f"Model tidak dikenal: {name}")
