import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    best_models,
    compute_metrics,
    coverage,
    diebold_mariano,
    directional_accuracy,
    mae,
    mape,
    mase,
    naive_scale,
    rmse,
)


def test_basic_metrics():
    y, yhat = np.array([100, 200, 300.0]), np.array([110, 190, 300.0])
    assert mae(y, yhat) == pytest.approx(20 / 3)
    assert rmse(y, yhat) == pytest.approx(np.sqrt(200 / 3))
    assert mape(y, yhat) == pytest.approx(100 * (0.1 + 0.05) / 3)
    assert naive_scale([1, 3, 2, 6]) == pytest.approx((2 + 1 + 4) / 3)
    assert mase(y, yhat, 10) == pytest.approx(mae(y, yhat) / 10)
    assert coverage(y, y - 5, y + 5) == 100.0
    assert directional_accuracy([110, 90], [120, 95], 100) == 100.0
    assert directional_accuracy([110, 90], [95, 95], 100) == 50.0


def _preds():
    rows = []
    for model, bias in (("naive", 10.0), ("good", 2.0)):
        for fold in (1, 2):
            for step in range(1, 31):
                y = 100 + step
                rows.append(dict(region="r", commodity="c", model=model, split="backtest", fold=fold, step=step,
                                 y_true=y, yhat=y + bias, lo80=y - 1, hi80=y + 1, lo95=y - 20, hi95=y + 20,
                                 scale=5.0, y_origin=100.0))
    return pd.DataFrame(rows)


def test_compute_metrics_and_relative_mase():
    m = compute_metrics(_preds(), [7, 30])
    assert set(m["horizon"]) == {7, 30}
    good = m[(m["model"] == "good") & (m["horizon"] == 7)].iloc[0]
    assert good["mae"] == pytest.approx(2.0)
    assert good["mase"] == pytest.approx(0.4)
    assert good["rel_mase"] == pytest.approx(0.2)
    assert good["coverage95"] == 100.0
    assert good["n_folds"] == 2
    assert best_models(m).iloc[0]["model"] == "good"


def test_diebold_mariano_detects_better_model():
    rng = np.random.default_rng(0)
    e_good = rng.normal(0, 1, 300)
    e_bad = rng.normal(0, 3, 300)
    stat, p = diebold_mariano(e_good, e_bad, h=1)
    assert stat < 0 and p < 0.01
    stat, p = diebold_mariano(e_good, e_good + rng.normal(0, 0.01, 300), h=1)
    assert p > 0.05
