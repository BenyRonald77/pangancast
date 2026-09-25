import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def price_series() -> pd.Series:
    """Deret harga sintetis 3 tahun: tren + musiman mingguan/tahunan + noise (deterministik)."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=3 * 365, freq="D")
    t = np.arange(len(idx))
    log_p = (np.log(30000) + 0.0002 * t + 0.03 * np.sin(2 * np.pi * t / 7)
             + 0.1 * np.sin(2 * np.pi * t / 365.25) + np.cumsum(rng.normal(0, 0.01, len(idx))))
    return pd.Series(np.exp(log_p), index=idx, name="test")
