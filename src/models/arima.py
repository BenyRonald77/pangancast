"""ARIMA/SARIMA dengan pemilihan orde otomatis (``pmdarima.auto_arima``, uji ADF, m=7)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

from src.models.base import Forecaster


def select_order(log_y: pd.Series, m: int = 7, seasonal: bool = True, max_p: int = 3, max_q: int = 3,
                 max_P: int = 1, max_Q: int = 1) -> tuple[tuple, tuple]:
    """Pilih orde (p,d,q)(P,D,Q,m) dengan auto_arima (stepwise, AIC).

    Orde differencing ``d`` ditentukan dengan uji ADF.
    """
    import pmdarima as pm

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = pm.auto_arima(
            log_y.to_numpy(),
            seasonal=seasonal, m=m if seasonal else 1,
            test="adf", seasonal_test="ocsb",
            max_p=max_p, max_q=max_q, max_P=max_P, max_Q=max_Q, max_d=2, max_D=1,
            stepwise=True, information_criterion="aic",
            suppress_warnings=True, error_action="ignore",
        )
    return tuple(model.order), tuple(model.seasonal_order)


class ArimaForecaster(Forecaster):
    """SARIMA pada log harga; forecast multi-langkah rekursif bawaan state space.

    Jika ``order`` tidak diberikan, orde dipilih dengan ``auto_arima`` pada
    ``select_train_days`` hari terakhir data train, lalu model di-fit ulang pada
    seluruh data train dengan orde tersebut.
    """

    name = "arima"

    def __init__(self, order=None, seasonal_order=None, m: int = 7, seasonal: bool = True,
                 max_p: int = 3, max_q: int = 3, max_P: int = 1, max_Q: int = 1,
                 select_train_days: int | None = 1095, **params):
        super().__init__(order=order, seasonal_order=seasonal_order, m=m, seasonal=seasonal, **params)
        self.order = tuple(order) if order else None
        self.seasonal_order = tuple(seasonal_order) if seasonal_order else None
        self.select_kw = dict(m=m, seasonal=seasonal, max_p=max_p, max_q=max_q, max_P=max_P, max_Q=max_Q)
        self.select_train_days = select_train_days

    def _fit(self, log_y: pd.Series) -> None:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        if self.order is None:
            sel = log_y.iloc[-self.select_train_days:] if self.select_train_days else log_y
            self.order, self.seasonal_order = select_order(sel, **self.select_kw)
        # Konstanta hanya untuk model tanpa differencing; drift pada log harga
        # harian sangat kecil sehingga diabaikan untuk model terdiferensiasi.
        d_total = self.order[1] + (self.seasonal_order[1] if self.seasonal_order else 0)
        trend = "c" if d_total == 0 else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_ = SARIMAX(
                log_y.to_numpy(), order=self.order,
                seasonal_order=self.seasonal_order or (0, 0, 0, 0), trend=trend,
                enforce_stationarity=False, enforce_invertibility=False,
            ).fit(disp=False, maxiter=200)

    def _predict(self, horizon, levels):
        fc = self.model_.get_forecast(steps=horizon)
        mean = np.asarray(fc.predicted_mean, dtype=float)
        se = np.asarray(fc.se_mean, dtype=float)
        bounds = {}
        for lv in levels:
            z = stats.norm.ppf(0.5 + lv / 2)
            p = int(round(lv * 100))
            bounds[f"lo{p}"], bounds[f"hi{p}"] = mean - z * se, mean + z * se
        return mean, bounds

    @property
    def spec(self) -> str:
        """Orde model dalam notasi SARIMA."""
        return f"SARIMA{self.order}x{self.seasonal_order}"
