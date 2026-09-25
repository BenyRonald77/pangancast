"""F-06/F-07 — Metrik evaluasi, agregasi per horizon, leaderboard, dan uji Diebold-Mariano."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

METRICS = ["mae", "rmse", "mape", "mase", "coverage80", "coverage95", "directional_acc"]


def mae(y, yhat) -> float:
    """Mean absolute error (Rp/kg)."""
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def rmse(y, yhat) -> float:
    """Root mean squared error (Rp/kg)."""
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def mape(y, yhat) -> float:
    """Mean absolute percentage error (%)."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(100 * np.mean(np.abs((y - yhat) / y)))


def naive_scale(y_train) -> float:
    """Penyebut MASE: MAE naive satu langkah (lag 1) pada data train."""
    y_train = np.asarray(y_train, float)
    return float(np.mean(np.abs(np.diff(y_train))))


def mase(y, yhat, scale: float) -> float:
    """Mean absolute scaled error; < 1 berarti lebih baik dari naive in-sample."""
    return mae(y, yhat) / scale if scale > 0 else float("nan")


def coverage(y, lo, hi) -> float:
    """Proporsi (%) nilai aktual yang jatuh di dalam prediction interval."""
    y = np.asarray(y, float)
    return float(100 * np.mean((y >= np.asarray(lo)) & (y <= np.asarray(hi))))


def directional_accuracy(y, yhat, y_origin: float) -> float:
    """Persentase arah perubahan (naik/turun/tetap) terhadap harga origin yang tertebak benar."""
    true_dir = np.sign(np.asarray(y, float) - y_origin)
    pred_dir = np.sign(np.round(np.asarray(yhat, float) - y_origin, 0))
    return float(100 * np.mean(true_dir == pred_dir))


def fold_metrics(g: pd.DataFrame) -> dict[str, float]:
    """Hitung semua metrik untuk satu fold (satu origin) pada langkah yang dipilih."""
    return {
        "mae": mae(g["y_true"], g["yhat"]),
        "rmse": rmse(g["y_true"], g["yhat"]),
        "mape": mape(g["y_true"], g["yhat"]),
        "mase": mase(g["y_true"], g["yhat"], g["scale"].iloc[0]),
        "coverage80": coverage(g["y_true"], g["lo80"], g["hi80"]),
        "coverage95": coverage(g["y_true"], g["lo95"], g["hi95"]),
        "directional_acc": directional_accuracy(g["y_true"], g["yhat"], g["y_origin"].iloc[0]),
    }


def compute_metrics(preds: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Metrik per region × komoditas × model × split × horizon, dirata-rata antar fold.

    Horizon ``h`` memakai langkah 1..h dari setiap forecast (sesuai rumus MASE PRD).

    Args:
        preds: prediksi backtest dengan kolom ``region, commodity, model, split,
            fold, step, y_true, yhat, lo80, hi80, lo95, hi95, scale, y_origin``.
    """
    keys = ["region", "commodity", "model", "split"]
    rows = []
    for h in horizons:
        sub = preds[preds["step"] <= h]
        per_fold = (
            sub.groupby(keys + ["fold"])
            .apply(lambda g: pd.Series(fold_metrics(g)), include_groups=False)
            .reset_index()
        )
        agg = per_fold.groupby(keys)[METRICS].mean().reset_index()
        agg["horizon"] = h
        agg["n_folds"] = per_fold.groupby(keys)["fold"].nunique().to_numpy()
        rows.append(agg)
    out = pd.concat(rows, ignore_index=True)
    # MASE relatif terhadap baseline naive pada fold & horizon yang sama:
    # < 1 berarti model mengalahkan naive (metrik keberhasilan PRD).
    naive = out[out["model"] == "naive"][["region", "commodity", "split", "horizon", "mase"]]
    out = out.merge(naive.rename(columns={"mase": "_naive_mase"}),
                    on=["region", "commodity", "split", "horizon"], how="left")
    naive_mase = out.pop("_naive_mase")
    # Bila naive sempurna (harga konstan selama test), rasio tak terdefinisi → NaN.
    out["rel_mase"] = np.where(naive_mase > 1e-9, out["mase"] / naive_mase.where(naive_mase > 1e-9), np.nan)
    return out


def leaderboard(metrics: pd.DataFrame, split: str = "backtest", horizon: int = 7) -> pd.DataFrame:
    """Peringkat model per deret (MASE terendah = pemenang) pada split/horizon tertentu."""
    m = metrics[(metrics["split"] == split) & (metrics["horizon"] == horizon)].copy()
    m["rank"] = m.groupby(["region", "commodity"])["mase"].rank(method="min")
    return m.sort_values(["region", "commodity", "rank"]).reset_index(drop=True)


def best_models(metrics: pd.DataFrame, horizon: int = 7,
                baselines: tuple[str, ...] = ("naive", "seasonal_naive")) -> pd.DataFrame:
    """Model terbaik per region × komoditas berdasarkan MASE backtest pada horizon 7.

    Pemenang dipilih dari model kandidat (bukan baseline); baseline naive menjadi
    patokan lewat kolom ``rel_mase`` (< 1 = mengalahkan naive).
    """
    cand = metrics[~metrics["model"].isin(baselines)]
    lb = leaderboard(cand, "backtest", horizon)
    return lb[lb["rank"] == 1].drop_duplicates(["region", "commodity"])[
        ["region", "commodity", "model", "mase", "rel_mase", "mape"]
    ].reset_index(drop=True)


def diebold_mariano(e1, e2, h: int = 1, power: int = 2) -> tuple[float, float]:
    """Uji Diebold-Mariano dengan koreksi Harvey-Leybourne-Newbold.

    H0: akurasi kedua forecast sama. Statistik negatif berarti model 1 lebih akurat.

    Args:
        e1, e2: error forecast (aktual − prediksi) yang sejajar.
        h: horizon forecast (untuk autokovarians lag < h).
        power: 2 = squared error loss, 1 = absolute error loss.

    Returns:
        (statistik DM, p-value dua sisi)
    """
    e1, e2 = np.asarray(e1, float), np.asarray(e2, float)
    d = np.abs(e1) ** power - np.abs(e2) ** power
    n = len(d)
    d_mean = d.mean()
    gamma = [np.sum((d[k:] - d_mean) * (d[: n - k] - d_mean)) / n for k in range(h)]
    var_d = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var_d <= 0:
        return float("nan"), float("nan")
    dm = d_mean / np.sqrt(var_d)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = float(dm * hln)
    p = float(2 * stats.t.sf(abs(stat), df=n - 1))
    return stat, p


def dm_tests(preds: pd.DataFrame, metrics: pd.DataFrame, horizon: int = 7) -> pd.DataFrame:
    """Uji DM per deret: pemenang resmi (model kandidat) vs pesaing terdekatnya.

    Pesaing = model lain dengan MASE terendah, termasuk baseline, sehingga uji ini
    menjawab apakah keunggulan pemenang signifikan secara statistik. Prediksi yang
    identik (mis. SARIMA berorde random walk vs naive) menghasilkan p-value NaN.
    """
    lb = leaderboard(metrics, "backtest", horizon)
    winners = best_models(metrics, horizon).set_index(["region", "commodity"])["model"]
    rows = []
    for (region, commodity), g in lb.groupby(["region", "commodity"]):
        if (region, commodity) not in winners.index:
            continue
        best = winners[(region, commodity)]
        others = g[g["model"] != best].sort_values("mase")
        if others.empty:
            continue
        second = others.iloc[0]["model"]
        p = preds[(preds["region"] == region) & (preds["commodity"] == commodity)
                  & (preds["split"] == "backtest") & (preds["step"] <= horizon)]
        a = p[p["model"] == best].set_index(["fold", "step"])
        b = p[p["model"] == second].set_index(["fold", "step"])
        idx = a.index.intersection(b.index)
        stat, pval = diebold_mariano(
            (a.loc[idx, "y_true"] - a.loc[idx, "yhat"]).to_numpy(),
            (b.loc[idx, "y_true"] - b.loc[idx, "yhat"]).to_numpy(),
            h=horizon,
        )
        rows.append({"region": region, "commodity": commodity, "best": best, "runner_up": second,
                     "horizon": horizon, "dm_stat": stat, "p_value": pval, "n": len(idx),
                     "significant_5pct": bool(pval < 0.05) if not np.isnan(pval) else False})
    return pd.DataFrame(rows)
