"""Antarmuka seragam untuk semua model forecasting.

Semua model dilatih pada **log harga** dan mengembalikan prediksi dalam Rupiah.
Interval dihitung di ruang log lalu ditransformasi balik dengan ``exp``.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy import stats

LEVELS = (0.8, 0.95)


def interval_cols(levels=LEVELS) -> list[str]:
    """Nama kolom interval, mis. ``lo80, hi80, lo95, hi95``."""
    cols = []
    for lv in levels:
        p = int(round(lv * 100))
        cols += [f"lo{p}", f"hi{p}"]
    return cols


def normal_bounds(mean: np.ndarray, sigma: np.ndarray, levels=LEVELS) -> dict[str, np.ndarray]:
    """Batas interval normal simetris di ruang log."""
    out = {}
    for lv in levels:
        z = stats.norm.ppf(0.5 + lv / 2)
        p = int(round(lv * 100))
        out[f"lo{p}"], out[f"hi{p}"] = mean - z * sigma, mean + z * sigma
    return out


def empirical_bounds(mean: np.ndarray, residuals: np.ndarray, levels=LEVELS) -> dict[str, np.ndarray]:
    """Batas interval dari kuantil residual empiris per langkah (split conformal).

    Args:
        mean: prediksi (log) per langkah, bentuk ``(h,)``.
        residuals: residual kalibrasi (aktual − prediksi, log), bentuk ``(n, h)``.
    """
    out = {}
    for lv in levels:
        p = int(round(lv * 100))
        lo_q = np.nanquantile(residuals, (1 - lv) / 2, axis=0)
        hi_q = np.nanquantile(residuals, (1 + lv) / 2, axis=0)
        out[f"lo{p}"], out[f"hi{p}"] = mean + lo_q, mean + hi_q
    return out


class Forecaster(ABC):
    """Kelas dasar. Subclass mengimplementasikan ``_fit`` dan ``_predict`` di ruang log."""

    name: str = "base"

    def __init__(self, **params):
        self.params = params
        self.last_date: pd.Timestamp | None = None
        self.fit_time_s = 0.0
        self.predict_time_s = 0.0

    def fit(self, y: pd.Series) -> Forecaster:
        """Latih model pada deret harga harian (Rupiah, index tanggal)."""
        if (y <= 0).any():
            raise ValueError("Harga harus positif untuk transformasi log")
        t0 = time.perf_counter()
        log_y = np.log(y.astype(float)).asfreq("D")
        self.last_date = log_y.index[-1]
        self._fit(log_y)
        self.fit_time_s = time.perf_counter() - t0
        return self

    def predict(self, horizon: int, levels=LEVELS) -> pd.DataFrame:
        """Forecast ``horizon`` hari ke depan.

        Returns:
            DataFrame index tanggal, kolom ``yhat`` + ``lo80, hi80, lo95, hi95`` (Rupiah).
        """
        if self.last_date is None:
            raise RuntimeError("Model belum di-fit")
        t0 = time.perf_counter()
        mean, bounds = self._predict(horizon, levels)
        self.predict_time_s = time.perf_counter() - t0
        idx = pd.date_range(self.last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
        out = pd.DataFrame({"yhat": np.exp(mean)}, index=idx)
        for col in interval_cols(levels):
            out[col] = np.exp(bounds[col])
        # Pastikan interval selalu mengapit prediksi titik.
        for lv in levels:
            p = int(round(lv * 100))
            out[f"lo{p}"] = np.minimum(out[f"lo{p}"], out["yhat"])
            out[f"hi{p}"] = np.maximum(out[f"hi{p}"], out["yhat"])
        out.index.name = "date"
        return out

    @abstractmethod
    def _fit(self, log_y: pd.Series) -> None: ...

    @abstractmethod
    def _predict(self, horizon: int, levels) -> tuple[np.ndarray, dict[str, np.ndarray]]: ...
