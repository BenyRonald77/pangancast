"""XGBoost dengan strategi *direct*: satu model per horizon (7, 14, 30).

* Model H7 menangani langkah 1–7, model H14 langkah 8–14, model H30 langkah 15–30.
  Langkah ``step`` menjadi fitur sehingga satu model bisa memprediksi seluruh
  langkah dalam rentangnya.
* Target: log return dari origin ke tanggal target.
* Fitur: lag/momentum 1–30 hari, rolling mean/std 7/14/30, kalender, hari raya.
* Interval: *split conformal* — model dilatih tanpa ``calibration_days`` hari
  terakhir, residual out-of-sample pada periode itu menjadi dasar kuantil per
  langkah, lalu model di-fit ulang pada seluruh data train.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import build_direct_dataset, feature_columns
from src.models.base import Forecaster, empirical_bounds

DEFAULT_BUCKETS = {7: range(1, 8), 14: range(8, 15), 30: range(15, 31)}
DEFAULT_LAGS = [1, 2, 3, 4, 5, 6, 7, 14, 21, 28, 30]
DEFAULT_WINDOWS = [7, 14, 30]


class XGBoostForecaster(Forecaster):
    """Gradient boosting tabular dengan model terpisah per horizon."""

    name = "xgboost"

    def __init__(self, n_estimators: int = 400, learning_rate: float = 0.05, max_depth: int = 4,
                 subsample: float = 0.8, colsample_bytree: float = 0.8, min_child_weight: float = 5,
                 reg_lambda: float = 1.0, calibration_days: int = 180, lags=None, windows=None,
                 buckets: dict | None = None, seed: int = 42, n_jobs: int = 4, **params):
        super().__init__(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
                         subsample=subsample, colsample_bytree=colsample_bytree,
                         min_child_weight=min_child_weight, reg_lambda=reg_lambda, **params)
        self.xgb_kw = dict(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
                           subsample=subsample, colsample_bytree=colsample_bytree,
                           min_child_weight=min_child_weight, reg_lambda=reg_lambda,
                           random_state=seed, n_jobs=n_jobs, tree_method="hist")
        self.calibration_days = calibration_days
        self.lags = list(lags or DEFAULT_LAGS)
        self.windows = list(windows or DEFAULT_WINDOWS)
        self.buckets = {int(k): list(v) for k, v in (buckets or DEFAULT_BUCKETS).items()}

    def _fit_buckets(self, log_y: pd.Series) -> dict:
        from xgboost import XGBRegressor

        models = {}
        for h, steps in self.buckets.items():
            data = build_direct_dataset(log_y, steps, self.lags, self.windows)
            self.features_ = feature_columns(data)
            m = XGBRegressor(**self.xgb_kw)
            m.fit(data[self.features_], data["target"])
            models[h] = m
        return models

    def _predict_path(self, models: dict, log_y: pd.Series, origins: pd.DatetimeIndex) -> np.ndarray:
        """Prediksi log return untuk setiap origin × langkah 1..max; bentuk ``(n_origin, H)``."""
        max_step = max(max(s) for s in self.buckets.values())
        out = np.full((len(origins), max_step), np.nan)
        for h, steps in self.buckets.items():
            data = build_direct_dataset(log_y, steps, self.lags, self.windows,
                                        origins=origins, with_target=False)
            pred = models[h].predict(data[self.features_])
            rows = pd.Index(origins).get_indexer(data["origin"])
            out[rows, data["step"].to_numpy() - 1] = pred
        return out

    def _fit(self, log_y: pd.Series) -> None:
        max_step = max(max(s) for s in self.buckets.values())
        # 1) Kalibrasi interval: latih tanpa periode kalibrasi, ukur residual di sana.
        cal = self.calibration_days
        self.residuals_ = None
        if cal and len(log_y) > cal + 400:
            train_part = log_y.iloc[:-cal]
            cal_models = self._fit_buckets(train_part)
            origins = log_y.index[-cal - 1:-max_step]
            pred = self._predict_path(cal_models, log_y, origins)
            actual = np.stack([
                log_y.reindex(origins + pd.Timedelta(days=k)).to_numpy() - log_y.reindex(origins).to_numpy()
                for k in range(1, max_step + 1)
            ], axis=1)
            self.residuals_ = actual - pred
        # 2) Model final pada seluruh data train.
        self.models_ = self._fit_buckets(log_y)
        self.history_ = log_y.iloc[-(max(self.lags + self.windows) + 5):]

    def _predict(self, horizon, levels):
        origin = self.history_.index[-1:]
        ret = self._predict_path(self.models_, self.history_, origin)[0][:horizon]
        mean = self.history_.iloc[-1] + ret
        if self.residuals_ is not None:
            bounds = empirical_bounds(mean, self.residuals_[:, :horizon], levels)
        else:  # data terlalu pendek: fallback random walk
            from src.models.base import normal_bounds

            sigma = self.history_.diff().std() * np.sqrt(np.arange(1, horizon + 1))
            bounds = normal_bounds(mean, sigma, levels)
        return mean, bounds
