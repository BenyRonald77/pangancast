"""F-02 — Pembersihan, imputasi, dan validasi skema data harga.

Aturan (PRD):

* Data PIHPS hanya berisi hari kerja. Deret di-reindex ke frekuensi harian;
  celah ≤ 3 hari berturut-turut (akhir pekan, libur) diisi forward-fill,
  celah lebih panjang memakai interpolasi linear. Semua nilai hasil pengisian
  ditandai ``is_imputed = True``.
* Outlier dicek dengan robust rolling z-score (window 30 hari). Hanya kesalahan
  input yang jelas yang dibuang: lonjakan satu titik yang ekstrem **dan**
  langsung kembali ke level semula. Lonjakan nyata saat hari raya (bertahan
  beberapa hari) tidak disentuh.
* Hasil disimpan ke ``data/processed/prices.parquet``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import get_logger, load_config, path
from src.ingest import load_raw

log = get_logger("clean")

SCHEMA = {
    "date": "datetime64[ns]",
    "region": "object",
    "commodity": "object",
    "price": "float64",
    "is_imputed": "bool",
}


def flag_input_errors(s: pd.Series, window: int = 30, z: float = 6.0, min_jump: float = 0.4) -> pd.Series:
    """Tandai titik yang kemungkinan besar salah input.

    Kriteria: |robust z| > ``z`` terhadap median bergulir, dan titik tersebut
    menyimpang > ``min_jump`` (relatif) dari *kedua* tetangganya dengan arah
    yang sama (spike satu hari yang langsung berbalik).

    Args:
        s: deret harga pada hari observasi saja (tanpa NaN).

    Returns:
        Series boolean, True = salah input.
    """
    med = s.rolling(window, center=True, min_periods=5).median()
    mad = (s - med).abs().rolling(window, center=True, min_periods=5).median()
    robust_z = (s - med) / (1.4826 * mad.replace(0, np.nan))
    prev, nxt = s.shift(1), s.shift(-1)
    jump_prev = (s - prev) / prev
    jump_next = (s - nxt) / nxt
    spike = ((jump_prev > min_jump) & (jump_next > min_jump)) | (
        (jump_prev < -min_jump / (1 + min_jump)) & (jump_next < -min_jump / (1 + min_jump))
    )
    return (robust_z.abs() > z).fillna(False) & spike.fillna(False)


def clean_series(obs: pd.Series, ffill_limit: int = 3, **outlier_kw) -> pd.DataFrame:
    """Bersihkan satu deret harga (index = tanggal observasi).

    Returns:
        DataFrame harian dengan kolom ``price`` dan ``is_imputed``.
    """
    obs = obs.dropna().sort_index()
    obs = obs[~obs.index.duplicated(keep="last")]
    errors = flag_input_errors(obs, **outlier_kw)
    if errors.any():
        log.info("  %d titik salah input dibuang: %s", errors.sum(),
                 ", ".join(d.strftime("%Y-%m-%d") for d in obs.index[errors][:5]))
    obs = obs[~errors]

    idx = pd.date_range(obs.index.min(), obs.index.max(), freq="D")
    daily = obs.reindex(idx)
    missing = daily.isna()

    # Panjang setiap celah berturut-turut.
    gap_id = (~missing).cumsum()
    gap_len = missing.groupby(gap_id).transform("sum")
    short = missing & (gap_len <= ffill_limit)

    filled = daily.copy()
    filled[short] = daily.ffill()[short]
    filled = filled.interpolate(method="linear", limit_area="inside")
    return pd.DataFrame({"price": filled.round(2), "is_imputed": missing.to_numpy()}, index=idx)


def validate(df: pd.DataFrame) -> None:
    """Validasi skema tabel ``prices``; raise ``ValueError`` bila tidak sesuai."""
    missing_cols = set(SCHEMA) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Kolom hilang: {missing_cols}")
    if df["price"].isna().any():
        raise ValueError("Masih ada harga kosong setelah imputasi")
    if (df["price"] <= 0).any():
        raise ValueError("Harga harus positif")
    if df.duplicated(["date", "region", "commodity"]).any():
        raise ValueError("Duplikat (date, region, commodity)")
    for (r, c), g in df.groupby(["region", "commodity"]):
        d = g["date"].sort_values().diff().dropna()
        if not (d == pd.Timedelta(days=1)).all():
            raise ValueError(f"Tanggal tidak kontinu harian untuk {r}/{c}")


def build_prices(raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """Bangun tabel ``prices`` bersih dari data mentah."""
    cfg = load_config()
    ccfg = cfg["cleaning"]
    raw = load_raw() if raw is None else raw
    frames = []
    for (region, commodity), g in raw.groupby(["region", "commodity"]):
        if commodity not in cfg["commodities"] or region not in cfg["regions"]:
            continue
        log.info("Bersihkan %s / %s (%d observasi)", region, commodity, len(g))
        out = clean_series(
            g.set_index("date")["price"],
            ffill_limit=ccfg["ffill_limit"],
            window=ccfg["outlier_window"],
            z=ccfg["outlier_z"],
            min_jump=ccfg["outlier_min_jump"],
        )
        # Rasio celah di luar akhir pekan (hari kerja yang kosong).
        weekday = out.index.dayofweek < 5
        gap_ratio = out.loc[weekday, "is_imputed"].mean()
        if gap_ratio > ccfg["max_gap_ratio"]:
            log.warning("  Celah hari kerja %.1f%% > %.0f%% untuk %s/%s — pertimbangkan provinsi lain",
                        100 * gap_ratio, 100 * ccfg["max_gap_ratio"], region, commodity)
        out = out.rename_axis("date").reset_index()
        out["region"], out["commodity"] = region, commodity
        frames.append(out)
    df = pd.concat(frames, ignore_index=True)[list(SCHEMA)]
    df = df.astype(SCHEMA).sort_values(["commodity", "region", "date"]).reset_index(drop=True)
    validate(df)
    return df


def load_prices() -> pd.DataFrame:
    """Baca tabel ``prices`` bersih dari Parquet."""
    return pd.read_parquet(path("processed"))


def get_series(prices: pd.DataFrame, region: str, commodity: str) -> pd.Series:
    """Ambil satu deret harga harian (index tanggal, frekuensi D)."""
    g = prices[(prices["region"] == region) & (prices["commodity"] == commodity)]
    s = g.set_index("date")["price"].sort_index()
    s.index.freq = "D"
    s.name = f"{region}/{commodity}"
    return s


def main() -> None:
    """Entry point CLI."""
    df = build_prices()
    out = path("processed")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    summary = df.groupby(["region", "commodity"]).agg(
        start=("date", "min"), end=("date", "max"), n=("price", "size"),
        imputed_pct=("is_imputed", "mean"),
    )
    summary["imputed_pct"] = (100 * summary["imputed_pct"]).round(1)
    log.info("Tersimpan %s (%d baris)\n%s", out, len(df), summary.to_string())


if __name__ == "__main__":
    main()
