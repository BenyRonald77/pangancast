"""Kalender hari raya Indonesia untuk fitur dan regresor Prophet.

Tanggal Idulfitri dan Iduladha memakai tabel manual (keputusan pemerintah/SKB),
karena tanggal hijriah hasil perhitungan library bisa bergeser satu hari.
Untuk tahun di luar tabel dipakai library ``holidays`` bila terpasang.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pandas as pd

# Hari pertama hari raya (Indonesia).
IDULFITRI = {
    2017: "2017-06-25", 2018: "2018-06-15", 2019: "2019-06-05", 2020: "2020-05-24",
    2021: "2021-05-13", 2022: "2022-05-02", 2023: "2023-04-22", 2024: "2024-04-10",
    2025: "2025-03-31", 2026: "2026-03-20", 2027: "2027-03-10", 2028: "2028-02-27",
}
IDULADHA = {
    2017: "2017-09-01", 2018: "2018-08-22", 2019: "2019-08-11", 2020: "2020-07-31",
    2021: "2021-07-20", 2022: "2022-07-10", 2023: "2023-06-29", 2024: "2024-06-17",
    2025: "2025-06-06", 2026: "2026-05-27", 2027: "2027-05-17", 2028: "2028-05-05",
}

HOLIDAY_NAMES = ("idulfitri", "iduladha", "natal", "tahun_baru")


def _from_library(year: int, keyword: str) -> pd.Timestamp | None:
    try:
        import holidays
    except ImportError:  # pragma: no cover
        return None
    for d, name in sorted(holidays.country_holidays("ID", years=year).items()):
        if keyword.lower() in name.lower():
            return pd.Timestamp(d)
    return None


@cache
def holiday_dates(start_year: int, end_year: int) -> pd.DataFrame:
    """Tanggal hari raya utama sebagai DataFrame ``holiday, date``."""
    rows = []
    for y in range(start_year, end_year + 1):
        fitri = IDULFITRI.get(y) or _from_library(y, "Eid al-Fitr")
        adha = IDULADHA.get(y) or _from_library(y, "Eid al-Adha")
        if fitri is not None:
            rows.append(("idulfitri", pd.Timestamp(fitri)))
        if adha is not None:
            rows.append(("iduladha", pd.Timestamp(adha)))
        rows.append(("natal", pd.Timestamp(f"{y}-12-25")))
        rows.append(("tahun_baru", pd.Timestamp(f"{y}-01-01")))
    return pd.DataFrame(rows, columns=["holiday", "date"])


def holiday_features(dates: pd.DatetimeIndex, window: tuple[int, int] = (-14, 7)) -> pd.DataFrame:
    """Fitur kalender hari raya untuk setiap tanggal.

    Untuk tiap hari raya dibuat dua kolom:

    * ``{nama}_win`` — 1 bila tanggal berada di H-14 s.d. H+7.
    * ``{nama}_dist`` — jarak (hari) ke hari raya terdekat, dipotong ke jendela,
      nilai di luar jendela = batas jendela + 1 (tidak bocor ke masa depan karena
      tanggal hari raya sudah diketahui jauh hari).
    """
    lo, hi = window
    dates = pd.DatetimeIndex(dates)
    hol = holiday_dates(dates.min().year - 1, dates.max().year + 1)
    out = {}
    for name in HOLIDAY_NAMES:
        hd = hol.loc[hol["holiday"] == name, "date"].to_numpy().astype("datetime64[D]").astype(np.int64)
        day = dates.to_numpy().astype("datetime64[D]").astype(np.int64)
        # selisih hari (tanggal - hari raya) ke semua hari raya, ambil yang paling dekat
        diff = day[:, None] - hd[None, :]
        nearest = diff[range(len(dates)), abs(diff).argmin(axis=1)]
        in_win = (nearest >= lo) & (nearest <= hi)
        out[f"{name}_win"] = in_win.astype(int)
        out[f"{name}_dist"] = pd.Series(nearest).clip(lo - 1, hi + 1).to_numpy()
    return pd.DataFrame(out, index=dates)


def prophet_holidays(start_year: int, end_year: int, window: tuple[int, int] = (-14, 7)) -> pd.DataFrame:
    """DataFrame hari raya dalam format Prophet (``holiday, ds, lower_window, upper_window``)."""
    hol = holiday_dates(start_year, end_year).rename(columns={"date": "ds"})
    hol["lower_window"] = window[0]
    hol["upper_window"] = window[1]
    return hol
