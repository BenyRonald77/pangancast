import numpy as np
import pandas as pd
import pytest

from src.clean import clean_series, flag_input_errors, validate


def _business_days(n=120, start="2024-01-01", value=50000.0):
    idx = pd.bdate_range(start, periods=n)
    rng = np.random.default_rng(1)
    return pd.Series(value * np.exp(np.cumsum(rng.normal(0, 0.005, n))), index=idx)


def test_weekends_forward_filled_and_flagged():
    s = _business_days()
    out = clean_series(s)
    assert out.index.freq == "D" or (out.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all()
    sat = out.index[out.index.dayofweek == 5][0]
    assert out.loc[sat, "is_imputed"]
    assert out.loc[sat, "price"] == pytest.approx(s[s.index < sat].iloc[-1], rel=1e-6)
    assert not out["price"].isna().any()
    # hari kerja hasil observasi tidak ditandai imputasi
    assert not out.loc[s.index, "is_imputed"].any()


def test_long_gap_uses_linear_interpolation():
    s = _business_days()
    gap = s.index[20:30]  # 10 hari kerja hilang → celah > 3 hari kalender
    s2 = s.drop(gap)
    out = clean_series(s2)
    before, after = s2[s2.index < gap[0]].iloc[-1], s2[s2.index > gap[-1]].iloc[0]
    mid = out.loc[gap[4], "price"]
    assert min(before, after) - 1 <= mid <= max(before, after) + 1
    assert out.loc[gap, "is_imputed"].all()


def test_single_day_spike_removed_but_sustained_jump_kept():
    s = _business_days(n=200)
    spike_day = s.index[100]
    s_spike = s.copy()
    s_spike[spike_day] = s[spike_day] * 1.8
    assert flag_input_errors(s_spike).loc[spike_day]

    s_level = s.copy()
    s_level.iloc[100:110] *= 1.8  # lonjakan hari raya yang bertahan beberapa hari
    assert not flag_input_errors(s_level).any()


def test_validate_rejects_bad_schema():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=3), "region": "nasional",
                       "commodity": "x", "price": [1.0, np.nan, 2.0], "is_imputed": False})
    with pytest.raises(ValueError):
        validate(df)
    df["price"] = [1.0, 2.0, 3.0]
    validate(df)
    with pytest.raises(ValueError):
        validate(df.drop(index=1))  # tanggal tidak kontinu
