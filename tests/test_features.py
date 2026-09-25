import numpy as np
import pandas as pd

from src.calendar_id import holiday_features
from src.features import build_direct_dataset, origin_features


def test_origin_features_have_no_future_leakage(price_series):
    """Mengubah data setelah tanggal t tidak boleh mengubah fitur pada tanggal t."""
    log_y = np.log(price_series)
    f1 = origin_features(log_y, [1, 7, 30], [7, 30])
    t = log_y.index[500]
    perturbed = log_y.copy()
    perturbed.loc[perturbed.index > t] += 5.0
    f2 = origin_features(perturbed, [1, 7, 30], [7, 30])
    pd.testing.assert_series_equal(f1.loc[t], f2.loc[t])


def test_direct_dataset_target_is_future_log_return(price_series):
    log_y = np.log(price_series)
    df = build_direct_dataset(log_y, [1, 7], [1, 7], [7])
    row = df[df["step"] == 7].iloc[0]
    origin = row["origin"]
    expected = log_y[origin + pd.Timedelta(days=7)] - log_y[origin]
    assert np.isclose(row["target"], expected)
    # tidak ada baris dengan target melewati akhir data
    assert (df["origin"] + pd.to_timedelta(df["step"], unit="D") <= log_y.index[-1]).all()


def test_holiday_window_flags():
    dates = pd.date_range("2024-03-20", "2024-04-25", freq="D")  # Idulfitri 2024-04-10
    hf = holiday_features(dates)
    assert hf.loc["2024-03-27", "idulfitri_win"] == 1  # H-14
    assert hf.loc["2024-03-26", "idulfitri_win"] == 0  # H-15
    assert hf.loc["2024-04-17", "idulfitri_win"] == 1  # H+7
    assert hf.loc["2024-04-18", "idulfitri_win"] == 0
    assert hf.loc["2024-04-10", "idulfitri_dist"] == 0
