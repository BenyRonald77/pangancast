"""Prophet: dekomposisi aditif dengan musiman mingguan/tahunan dan regresor hari raya."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.calendar_id import prophet_holidays
from src.models.base import Forecaster

for _name in ("prophet", "cmdstanpy"):
    logging.getLogger(_name).setLevel(logging.WARNING)


class ProphetForecaster(Forecaster):
    """Prophet pada log harga.

    Hari raya (Idulfitri, Iduladha, Natal, Tahun Baru) dimasukkan dengan jendela
    H-14 s.d. H+7. Interval 80%/95% diambil dari kuantil *predictive samples*
    sehingga cukup satu kali fit untuk kedua level.
    """

    name = "prophet"

    def __init__(self, changepoint_prior_scale: float = 0.05, seasonality_prior_scale: float = 10.0,
                 holidays_prior_scale: float = 10.0, yearly_seasonality: bool = True,
                 weekly_seasonality: bool = True, uncertainty_samples: int = 300,
                 holiday_window: tuple[int, int] = (-14, 7), seed: int = 42, **params):
        super().__init__(changepoint_prior_scale=changepoint_prior_scale,
                         seasonality_prior_scale=seasonality_prior_scale,
                         holidays_prior_scale=holidays_prior_scale, **params)
        self.kw = dict(
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            holidays_prior_scale=holidays_prior_scale,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=False,
            uncertainty_samples=uncertainty_samples,
        )
        self.holiday_window = tuple(holiday_window)
        self.seed = seed

    def _fit(self, log_y: pd.Series) -> None:
        from prophet import Prophet

        years = (log_y.index.min().year - 1, log_y.index.max().year + 2)
        self.model_ = Prophet(holidays=prophet_holidays(*years, window=self.holiday_window), **self.kw)
        df = pd.DataFrame({"ds": log_y.index, "y": log_y.to_numpy()})
        self.model_.fit(df)

    def _predict(self, horizon, levels):
        future = pd.DataFrame(
            {"ds": pd.date_range(self.last_date + pd.Timedelta(days=1), periods=horizon, freq="D")}
        )
        mean = self.model_.predict(future)["yhat"].to_numpy()
        np.random.seed(self.seed)  # sampling ketidakpastian Prophet memakai numpy global RNG
        samples = self.model_.predictive_samples(future)["yhat"]  # (horizon, n_samples)
        bounds = {}
        for lv in levels:
            p = int(round(lv * 100))
            bounds[f"lo{p}"] = np.quantile(samples, (1 - lv) / 2, axis=1)
            bounds[f"hi{p}"] = np.quantile(samples, (1 + lv) / 2, axis=1)
        return mean, bounds
