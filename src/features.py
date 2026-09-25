"""Feature engineering untuk model tabular (XGBoost).

Semua fitur dihitung dari log harga **pada atau sebelum** tanggal origin ``t``
(``shift`` positif saja), sehingga tidak ada kebocoran data masa depan.
Fitur kalender untuk tanggal target ``t + k`` boleh dipakai karena kalender
(hari, bulan, tanggal hari raya) sudah diketahui sebelumnya.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.calendar_id import holiday_features


def origin_features(log_y: pd.Series, lags: list[int], windows: list[int]) -> pd.DataFrame:
    """Fitur per tanggal origin dari log harga.

    * ``ret_{l}``   : log p_t − log p_{t−l} (momentum)
    * ``rm_{w}``    : rata-rata bergulir log harga w hari − log p_t (deviasi dari tren)
    * ``rs_{w}``    : std bergulir return harian w hari (volatilitas)
    * ``level``     : log p_t (level harga)
    """
    f = {"level": log_y}
    for lag in lags:
        f[f"ret_{lag}"] = log_y - log_y.shift(lag)
    ret1 = log_y.diff()
    for w in windows:
        f[f"rm_{w}"] = log_y.rolling(w, min_periods=w).mean() - log_y
        f[f"rs_{w}"] = ret1.rolling(w, min_periods=w).std()
    return pd.DataFrame(f, index=log_y.index)


def calendar_features(dates: pd.DatetimeIndex, prefix: str = "") -> pd.DataFrame:
    """Fitur kalender: hari dalam minggu, bulan, hari dalam tahun (siklik), dan hari raya."""
    dates = pd.DatetimeIndex(dates)
    doy = dates.dayofyear.to_numpy()
    cal = pd.DataFrame(
        {
            "dow": dates.dayofweek,
            "month": dates.month,
            "doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        },
        index=dates,
    )
    cal = pd.concat([cal, holiday_features(dates)], axis=1)
    return cal.add_prefix(prefix)


def build_direct_dataset(
    log_y: pd.Series,
    steps: list[int],
    lags: list[int],
    windows: list[int],
    origins: pd.DatetimeIndex | None = None,
    with_target: bool = True,
) -> pd.DataFrame:
    """Dataset *direct* bertumpuk: satu baris per (origin, step).

    Target = ``log p_{t+step} − log p_t``. Fitur = fitur origin + ``step`` +
    kalender tanggal target.

    Args:
        log_y: log harga harian.
        steps: langkah ke depan yang ditangani model ini.
        origins: tanggal origin; default semua tanggal dengan fitur lengkap.
        with_target: False saat inferensi (target di masa depan belum ada).
    """
    feats = origin_features(log_y, lags, windows).dropna()
    if origins is not None:
        feats = feats.loc[feats.index.intersection(origins)]
    blocks = []
    for k in steps:
        target_dates = feats.index + pd.Timedelta(days=k)
        b = feats.copy()
        b["step"] = k
        b["origin"] = feats.index
        cal = calendar_features(target_dates, prefix="tgt_")
        b = pd.concat([b.reset_index(drop=True), cal.reset_index(drop=True)], axis=1)
        if with_target:
            future = log_y.reindex(target_dates).to_numpy()
            b["target"] = future - feats["level"].to_numpy()
        blocks.append(b)
    df = pd.concat(blocks, ignore_index=True)
    if with_target:
        df = df.dropna(subset=["target"])
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Kolom fitur (tanpa kolom bantu)."""
    return [c for c in df.columns if c not in ("origin", "target")]
