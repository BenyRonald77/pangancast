"""F-06 — Rolling-origin backtest (expanding window) dengan holdout akhir.

Tata letak waktu untuk satu deret (default config)::

    |--------------- train fold 1 ---------------|test 30|
    |------------------ train fold 2 ------------------|test 30|
    ...                                                   (12 fold backtest)
                                                          |-- holdout 90 hari --|
                                                          3 origin × 30 hari

* Semua model memakai fold, horizon, dan data yang sama persis.
* Tuning hanya memakai fold ``backtest``; fold ``holdout`` dipakai sekali di akhir.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.evaluate import naive_scale
from src.models import get_model


@dataclass(frozen=True)
class Fold:
    """Satu origin forecast."""

    fold: int
    split: str  # "backtest" atau "holdout"
    train_end: pd.Timestamp  # tanggal terakhir data train (origin)
    test_end: pd.Timestamp


def make_folds(dates: pd.DatetimeIndex, n_folds: int = 12, test_size: int = 30,
               holdout_days: int = 90) -> list[Fold]:
    """Buat daftar fold rolling-origin + holdout.

    Args:
        dates: index harian deret (kontinu).
        n_folds: jumlah fold backtest.
        test_size: panjang periode test per fold (= horizon maksimum).
        holdout_days: panjang holdout akhir; dibagi menjadi origin ber-jarak ``test_size``.
    """
    dates = pd.DatetimeIndex(dates)
    day = pd.Timedelta(days=1)
    holdout_start = dates[-holdout_days]
    folds = []
    for i in range(n_folds):
        test_start = holdout_start - (n_folds - i) * test_size * day
        folds.append(Fold(i + 1, "backtest", test_start - day, test_start + (test_size - 1) * day))
    for j in range(holdout_days // test_size):
        test_start = holdout_start + j * test_size * day
        folds.append(Fold(n_folds + j + 1, "holdout", test_start - day, test_start + (test_size - 1) * day))
    if folds[0].train_end < dates[0] + pd.Timedelta(days=365):
        raise ValueError("Data train fold pertama kurang dari 1 tahun")
    return folds


def run_folds(y: pd.Series, model_name: str, params: dict, folds: list[Fold], horizon: int = 30,
              seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """Latih dan evaluasi satu model pada semua fold.

    Untuk SARIMA, orde dipilih sekali pada train fold pertama lalu dipakai ulang
    di semua fold (konsisten dan jauh lebih cepat); parameter tetap di-fit ulang.

    Returns:
        (prediksi per fold × langkah, info waktu/konfigurasi)
    """
    params = dict(params or {})
    rows, fit_times, pred_times = [], [], []
    info: dict = {}
    for f in folds:
        train = y.loc[: f.train_end]
        test = y.loc[f.train_end + pd.Timedelta(days=1): f.test_end].iloc[:horizon]
        model = get_model(model_name, params, seed=seed)
        model.fit(train)
        if model_name == "arima" and not params.get("order"):
            params["order"], params["seasonal_order"] = model.order, model.seasonal_order
            info["spec"] = model.spec
        fc = model.predict(len(test))
        fit_times.append(model.fit_time_s)
        pred_times.append(model.predict_time_s)
        df = fc.reset_index()
        df["y_true"] = test.to_numpy()
        df["step"] = np.arange(1, len(test) + 1)
        df["fold"], df["split"] = f.fold, f.split
        df["origin"] = f.train_end
        df["y_origin"] = float(train.iloc[-1])
        df["scale"] = naive_scale(train.to_numpy())
        rows.append(df)
    info.update(
        train_time_s=float(np.mean(fit_times)),
        infer_time_s=float(np.mean(pred_times)),
        params={k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()},
    )
    return pd.concat(rows, ignore_index=True), info


def timed_final_fit(y: pd.Series, model_name: str, params: dict, horizon: int, seed: int = 42):
    """Fit pada seluruh data dan forecast ``horizon`` hari ke depan (untuk dashboard)."""
    t0 = time.perf_counter()
    model = get_model(model_name, params, seed=seed).fit(y)
    fc = model.predict(horizon)
    return model, fc, time.perf_counter() - t0
