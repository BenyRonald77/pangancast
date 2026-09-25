"""F-05 — Tuning hyperparameter dengan Optuna.

* Objektif: rata-rata MASE horizon 7 pada ``folds_used`` fold backtest terakhir,
  dirata-rata atas deret representatif (PRD: tuning pada 2 komoditas lalu
  diterapkan ke semua). Holdout tidak pernah disentuh.
* SARIMA tidak di-tuning di sini: ordenya dipilih ``auto_arima`` (AIC) per deret.
* Hasil ditulis ke ``configs/best_params.yaml`` dan otomatis dipakai ``make train``.

Contoh::

    python -m src.tune                         # xgboost, prophet, lstm
    python -m src.tune --models xgboost --trials 50
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import optuna
import pandas as pd
import yaml

from src.backtest import make_folds, run_folds
from src.clean import get_series, load_prices
from src.config import get_logger, load_config, path, set_seed
from src.evaluate import compute_metrics

log = get_logger("tune")

# Jumlah trial default per model (LSTM paling mahal).
TRIAL_SCALE = {"xgboost": 1.0, "prophet": 1.0, "lstm": 0.4}


def search_space(model: str, trial: optuna.Trial) -> dict:
    """Ruang pencarian hyperparameter per model."""
    if model == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 20, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10, log=True),
        }
    if model == "prophet":
        return {
            "changepoint_prior_scale": trial.suggest_float("changepoint_prior_scale", 0.001, 0.5, log=True),
            "seasonality_prior_scale": trial.suggest_float("seasonality_prior_scale", 0.01, 10, log=True),
            "holidays_prior_scale": trial.suggest_float("holidays_prior_scale", 0.01, 10, log=True),
        }
    if model == "lstm":
        return {
            "window": trial.suggest_categorical("window", [30, 60, 90]),
            "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
            "num_layers": trial.suggest_int("num_layers", 1, 2),
            "dropout": trial.suggest_float("dropout", 0.0, 0.4),
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        }
    raise ValueError(f"Tidak ada ruang tuning untuk {model}")


def objective_factory(model: str, series: list[tuple[pd.Series, list]], base_params: dict, seed: int):
    """Objektif Optuna: rata-rata MASE H7 pada fold tuning semua deret representatif."""

    def objective(trial: optuna.Trial) -> float:
        params = {**base_params, **search_space(model, trial)}
        if model == "xgboost":
            params["calibration_days"] = 0  # interval tidak dibutuhkan untuk objektif
        scores = []
        for y, folds in series:
            preds, _ = run_folds(y, model, params, folds, horizon=7, seed=seed)
            preds = preds.assign(region="r", commodity="c", model=model)
            m = compute_metrics(preds, [7])
            scores.append(float(m["mase"].mean()))
        return float(np.mean(scores))

    return objective


def tune(models: list[str], n_trials: int | None = None) -> dict:
    """Jalankan tuning dan simpan hasil ke ``configs/best_params.yaml``."""
    warnings.filterwarnings("ignore")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cfg = load_config()
    seed = set_seed(cfg["seed"])
    tcfg, bcfg = cfg["tuning"], cfg["backtest"]
    prices = load_prices()

    series = []
    for region, commodity in tcfg["representative"]:
        y = get_series(prices, region, commodity)
        folds = [f for f in make_folds(y.index, bcfg["n_folds"], bcfg["test_size"], bcfg["holdout_days"])
                 if f.split == "backtest"][-tcfg["folds_used"]:]
        series.append((y, folds))

    best_all = {}
    for model in models:
        trials = n_trials or max(5, int(round(tcfg["n_trials"] * TRIAL_SCALE.get(model, 1.0))))
        base = {k: v for k, v in (cfg["models"].get(model) or {}).items()}
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed),
                                    study_name=f"pangancast-{model}")
        # Trial pertama = parameter default config, agar hasil tuning tidak lebih buruk dari default.
        study.enqueue_trial({k: base[k] for k in _space_keys(model) if k in base})
        log.info("Tuning %s: %d trial", model, trials)
        study.optimize(objective_factory(model, series, base, seed), n_trials=trials, show_progress_bar=False)
        log.info("  terbaik MASE H7 = %.4f  %s", study.best_value, study.best_params)
        best_all[model] = {k: (float(v) if isinstance(v, np.floating) else v)
                           for k, v in study.best_params.items()}
        _save_best(model, best_all[model], study.best_value)
        out = path("outputs") / "tuning"
        out.mkdir(parents=True, exist_ok=True)
        study.trials_dataframe().to_csv(out / f"trials_{model}.csv", index=False)
    load_config.cache_clear()
    return best_all


def _save_best(model: str, params: dict, score: float) -> None:
    """Baca-ubah-tulis ``best_params.yaml`` per model (aman untuk tuning paralel per model)."""
    best_file = path("best_params")
    data = yaml.safe_load(best_file.read_text(encoding="utf-8")) if best_file.exists() else None
    data = data or {}
    data[model] = params
    best_file.write_text(
        "# Dihasilkan otomatis oleh `make tune` — menimpa parameter default di config.yaml\n"
        + yaml.safe_dump(data, sort_keys=True, allow_unicode=True),
        encoding="utf-8",
    )
    log.info("  %s: MASE H7 tuning = %.4f → %s", model, score, best_file)


def _space_keys(model: str) -> list[str]:
    keys = {
        "xgboost": ["n_estimators", "learning_rate", "max_depth", "subsample", "colsample_bytree",
                    "min_child_weight", "reg_lambda"],
        "prophet": ["changepoint_prior_scale", "seasonality_prior_scale", "holidays_prior_scale"],
        "lstm": ["window", "hidden_size", "num_layers", "dropout", "lr"],
    }
    return keys[model]


def main() -> None:
    """Entry point CLI."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=["xgboost", "prophet", "lstm"])
    ap.add_argument("--trials", type=int, help="jumlah trial (default dari config)")
    args = ap.parse_args()
    tune(args.models, args.trials)


if __name__ == "__main__":
    main()
