import numpy as np
import pandas as pd
import pytest

from src.backtest import make_folds, run_folds
from src.models import get_model
from src.models.lstm import make_windows


def _check_forecast(fc: pd.DataFrame, horizon: int, last_date: pd.Timestamp):
    assert len(fc) == horizon
    assert fc.index[0] == last_date + pd.Timedelta(days=1)
    assert (fc["yhat"] > 0).all()
    assert (fc["lo95"] <= fc["lo80"]).all() and (fc["lo80"] <= fc["yhat"]).all()
    assert (fc["yhat"] <= fc["hi80"]).all() and (fc["hi80"] <= fc["hi95"]).all()


def test_naive_models(price_series):
    fc = get_model("naive").fit(price_series).predict(14)
    _check_forecast(fc, 14, price_series.index[-1])
    assert np.allclose(fc["yhat"], price_series.iloc[-1])
    # interval melebar seiring horizon
    assert (np.diff(fc["hi95"] - fc["lo95"]) > 0).all()

    sn = get_model("seasonal_naive").fit(price_series).predict(14)
    assert np.allclose(sn["yhat"].to_numpy()[:7], price_series.iloc[-7:].to_numpy())
    assert np.allclose(sn["yhat"].to_numpy()[7:], price_series.iloc[-7:].to_numpy())


def test_xgboost_direct_forecast(price_series):
    m = get_model("xgboost", {"n_estimators": 30, "calibration_days": 120, "n_jobs": 1})
    fc = m.fit(price_series).predict(30)
    _check_forecast(fc, 30, price_series.index[-1])
    assert m.residuals_.shape[1] == 30
    assert set(m.models_) == {7, 14, 30}


def test_lstm_windows_and_forecast(price_series):
    x = np.column_stack([np.arange(100, dtype=float), np.zeros(100)])
    X, Y = make_windows(x, window=10, horizon=5)
    assert X.shape == (86, 10, 2) and Y.shape == (86, 5)
    assert np.allclose(Y[0], [1, 2, 3, 4, 5])  # selisih terhadap nilai terakhir jendela

    m = get_model("lstm", {"window": 30, "hidden_size": 8, "max_epochs": 3, "patience": 2})
    fc = m.fit(price_series).predict(30)
    _check_forecast(fc, 30, price_series.index[-1])


def test_models_can_be_saved_and_loaded(price_series, tmp_path):
    import joblib

    for name, params in [("naive", {}), ("xgboost", {"n_estimators": 10, "calibration_days": 0}),
                         ("lstm", {"window": 30, "hidden_size": 8, "max_epochs": 1, "patience": 1})]:
        m = get_model(name, params).fit(price_series)
        f = tmp_path / f"{name}.joblib"
        joblib.dump(m, f)
        pd.testing.assert_frame_equal(joblib.load(f).predict(7), m.predict(7))


def test_lstm_is_reproducible(price_series):
    params = {"window": 30, "hidden_size": 8, "max_epochs": 2, "patience": 2}
    a = get_model("lstm", params, seed=7).fit(price_series).predict(7)
    b = get_model("lstm", params, seed=7).fit(price_series).predict(7)
    pd.testing.assert_frame_equal(a, b)


def test_make_folds_layout(price_series):
    folds = make_folds(price_series.index, n_folds=4, test_size=30, holdout_days=90)
    assert [f.split for f in folds] == ["backtest"] * 4 + ["holdout"] * 3
    assert folds[-1].test_end == price_series.index[-1]
    # fold berurutan, tidak tumpang tindih, dan train selalu sebelum test
    for a, b in zip(folds, folds[1:]):
        assert b.train_end == a.test_end
    with pytest.raises(ValueError):
        make_folds(price_series.index[:400], n_folds=12)


def test_run_folds_output(price_series):
    folds = make_folds(price_series.index, n_folds=2, test_size=30, holdout_days=30)
    preds, info = run_folds(price_series, "naive", {}, folds, horizon=30)
    assert len(preds) == 3 * 30
    assert {"y_true", "yhat", "scale", "y_origin", "split"} <= set(preds.columns)
    # y_true selaras dengan tanggal
    r = preds.iloc[5]
    assert r["y_true"] == pytest.approx(price_series[r["date"]])
    assert info["train_time_s"] >= 0


@pytest.mark.slow
def test_arima_and_prophet(price_series):
    y = price_series.iloc[-500:]
    arima = get_model("arima", {"order": (1, 1, 1), "seasonal_order": (0, 0, 0, 0)})
    _check_forecast(arima.fit(y).predict(7), 7, y.index[-1])
    prophet = get_model("prophet", {"uncertainty_samples": 50})
    _check_forecast(prophet.fit(y).predict(7), 7, y.index[-1])
