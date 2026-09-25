"""F-04/F-06/F-08 — Latih semua model untuk semua deret, evaluasi, dan simpan artefak.

Keluaran (sumber tunggal untuk dashboard dan README):

* ``outputs/metrics.csv``               metrik per region × komoditas × model × split × horizon
* ``outputs/backtest_predictions.parquet`` prediksi setiap fold (backtest + holdout)
* ``outputs/forecast.parquet``          forecast 30 hari ke depan dari seluruh data, semua model
* ``outputs/best_models.csv``           pemenang per deret (MASE backtest horizon 7)
* ``outputs/dm_test.csv``               uji Diebold-Mariano pemenang vs runner-up
* ``outputs/run_info.json``             metadata run (tanggal data, versi, parameter)
* ``models/*.joblib``                   model terlatih pada seluruh data

Contoh::

    python -m src.train                       # semua model, semua deret
    python -m src.train --models naive xgboost --series nasional/cabai_rawit_merah
"""

from __future__ import annotations

import argparse
import json
import platform
import time
import warnings

import joblib
import pandas as pd
from joblib import Parallel, delayed

from src.backtest import make_folds, run_folds, timed_final_fit
from src.clean import get_series, load_prices
from src.config import get_logger, load_config, path, set_seed
from src.evaluate import best_models, compute_metrics, dm_tests

log = get_logger("train")


def _job(y: pd.Series, region: str, commodity: str, model_name: str, params: dict, folds, cfg: dict,
         save_model: bool):
    """Satu pekerjaan: backtest + holdout + fit final untuk satu deret × model."""
    warnings.filterwarnings("ignore")
    set_seed(cfg["seed"])
    horizon = cfg["backtest"]["max_horizon"]
    t0 = time.perf_counter()
    preds, info = run_folds(y, model_name, params, folds, horizon=horizon, seed=cfg["seed"])
    final_params = {**params, **info.get("params", {})}
    model, fc, final_time = timed_final_fit(y, model_name, final_params, horizon, seed=cfg["seed"])
    tag = dict(region=region, commodity=commodity, model=model_name)
    preds = preds.assign(**tag)
    fc = fc.reset_index().assign(**tag, step=range(1, horizon + 1), origin=y.index[-1])
    if save_model:
        out = path("models") / f"{region}__{commodity}__{model_name}.joblib"
        joblib.dump(model, out, compress=3)
    info.update(tag, final_fit_s=final_time, total_s=time.perf_counter() - t0)
    info.pop("params", None)
    info["spec"] = getattr(model, "spec", None) or info.get("spec")
    return preds, fc, info


def run(models: list[str] | None = None, series: list[str] | None = None, n_jobs: int = 6,
        save_models: bool = True) -> pd.DataFrame:
    """Jalankan seluruh pipeline training dan evaluasi."""
    cfg = load_config()
    set_seed(cfg["seed"])
    bcfg = cfg["backtest"]
    prices = load_prices()
    models = models or cfg["models"]["enabled"]
    path("models").mkdir(parents=True, exist_ok=True)
    path("outputs").mkdir(parents=True, exist_ok=True)

    tasks = []
    for region in cfg["regions"]:
        for commodity in cfg["commodities"]:
            if series and f"{region}/{commodity}" not in series:
                continue
            y = get_series(prices, region, commodity)
            folds = make_folds(y.index, bcfg["n_folds"], bcfg["test_size"], bcfg["holdout_days"])
            for m in models:
                params = dict(cfg["models"].get(m) or {})
                tasks.append((y, region, commodity, m, params, folds))
    log.info("Menjalankan %d pekerjaan (%d model) dengan n_jobs=%d", len(tasks), len(models), n_jobs)

    # Model lambat dijalankan duluan agar paralelisme efisien.
    cost = {"lstm": 0, "arima": 1, "prophet": 2, "xgboost": 3}
    tasks.sort(key=lambda t: cost.get(t[3], 9))
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_job)(*t, cfg, save_models) for t in tasks
    )
    preds = pd.concat([r[0] for r in results], ignore_index=True)
    forecast = pd.concat([r[1] for r in results], ignore_index=True)
    info = pd.DataFrame([r[2] for r in results])

    out = path("outputs")
    _merge_write(preds, out / "backtest_predictions.parquet", models, series)
    _merge_write(forecast, out / "forecast.parquet", models, series)
    _merge_write(info, out / "model_info.csv", models, series)
    all_metrics, best = evaluate_saved()

    run_info = {
        "run_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "data_start": str(prices["date"].min().date()),
        "data_end": str(prices["date"].max().date()),
        "n_series": int(prices.groupby(["region", "commodity"]).ngroups),
        "folds": bcfg,
        "seed": cfg["seed"],
        "python": platform.python_version(),
        "models": {m: cfg["models"].get(m) for m in cfg["models"]["enabled"]},
    }
    (out / "run_info.json").write_text(json.dumps(run_info, indent=2, default=str), encoding="utf-8")
    log.info("Pemenang per deret (MASE backtest H7):\n%s", best.to_string(index=False))
    return all_metrics


def evaluate_saved() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hitung ulang metrik, pemenang, dan uji DM dari semua prediksi tersimpan di ``outputs/``.

    Selalu memakai seluruh prediksi (bukan hanya run terakhir) agar MASE relatif
    terhadap naive tetap terdefinisi pada run parsial.
    """
    cfg = load_config()
    out = path("outputs")
    preds = pd.read_parquet(out / "backtest_predictions.parquet")
    info = pd.read_csv(out / "model_info.csv")
    metrics = compute_metrics(preds, cfg["backtest"]["horizons"]).merge(
        info[["region", "commodity", "model", "train_time_s", "infer_time_s"]],
        on=["region", "commodity", "model"], how="left",
    )
    metrics.to_csv(out / "metrics.csv", index=False)
    best = best_models(metrics, horizon=7)
    best.to_csv(out / "best_models.csv", index=False)
    dm_tests(preds, metrics, horizon=7).to_csv(out / "dm_test.csv", index=False)
    return metrics, best


def _merge_write(df: pd.DataFrame, file, models, series) -> None:
    """Tulis hasil; bila run parsial (subset model/deret), gabungkan dengan hasil lama."""
    partial = bool(series) or set(models) != set(load_config()["models"]["enabled"])
    if partial and file.exists():
        old = pd.read_parquet(file) if file.suffix == ".parquet" else pd.read_csv(file)
        mask = old["model"].isin(models)
        if series:
            mask &= (old["region"] + "/" + old["commodity"]).isin(series)
        df = pd.concat([old[~mask], df], ignore_index=True)
    if file.suffix == ".parquet":
        df.to_parquet(file, index=False)
    else:
        df.to_csv(file, index=False)


def main() -> None:
    """Entry point CLI."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", help="subset model (default: semua di config)")
    ap.add_argument("--series", nargs="*", help="subset deret, format region/komoditas")
    ap.add_argument("--n-jobs", type=int, default=6)
    ap.add_argument("--no-save-models", action="store_true")
    ap.add_argument("--evaluate-only", action="store_true", help="hitung ulang metrik dari prediksi tersimpan")
    args = ap.parse_args()
    if args.evaluate_only:
        _, best = evaluate_saved()
        log.info("Metrik dihitung ulang. Pemenang:\n%s", best.to_string(index=False))
        return
    run(args.models, args.series, args.n_jobs, save_models=not args.no_save_models)


if __name__ == "__main__":
    main()
