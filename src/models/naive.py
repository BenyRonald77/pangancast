"""Baseline naive (harga besok = harga hari ini) dan seasonal naive (lag 7)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.base import Forecaster, normal_bounds


class NaiveForecaster(Forecaster):
    """Random walk: semua langkah = nilai terakhir.

    Interval: σ return harian log × √k (varians random walk).
    """

    name = "naive"

    def _fit(self, log_y: pd.Series) -> None:
        self.last_ = float(log_y.iloc[-1])
        self.sigma_ = float(log_y.diff().dropna().std())

    def _predict(self, horizon, levels):
        mean = np.full(horizon, self.last_)
        sigma = self.sigma_ * np.sqrt(np.arange(1, horizon + 1))
        return mean, normal_bounds(mean, sigma, levels)


class SeasonalNaiveForecaster(Forecaster):
    """Seasonal naive mingguan: nilai langkah k = nilai 7 hari sebelumnya pada hari yang sama.

    Interval: σ selisih musiman (lag m) × √(jumlah siklus ke depan).
    """

    name = "seasonal_naive"

    def __init__(self, m: int = 7, **params):
        super().__init__(m=m, **params)
        self.m = m

    def _fit(self, log_y: pd.Series) -> None:
        self.season_ = log_y.iloc[-self.m:].to_numpy(dtype=float)
        self.sigma_ = float((log_y - log_y.shift(self.m)).dropna().std())

    def _predict(self, horizon, levels):
        k = np.arange(horizon)
        mean = self.season_[k % self.m]
        sigma = self.sigma_ * np.sqrt(k // self.m + 1)
        return mean, normal_bounds(mean, sigma, levels)
