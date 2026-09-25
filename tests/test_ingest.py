import math

import pandas as pd

from src.ingest import grid_to_long, parse_price

REGIONS = {"nasional": "Semua Provinsi", "diy": "DI Yogyakarta"}


def test_parse_price_formats():
    assert parse_price("67,900") == 67900.0
    assert parse_price("1,250,000") == 1250000.0
    assert parse_price(45000) == 45000.0
    for bad in ("-", "", None, "0", "abc"):
        assert math.isnan(parse_price(bad))


def test_grid_to_long_maps_regions_and_dates():
    rows = [
        {"no": "I", "name": "Semua Provinsi", "level": 0, "01/03/2024": "67,900", "04/03/2024": "-"},
        {"no": "II", "name": "DI Yogyakarta", "level": 1, "01/03/2024": "68,750", "04/03/2024": "61,250"},
        {"no": "III", "name": "Jawa Tengah", "level": 1, "01/03/2024": "60,000"},
    ]
    df = grid_to_long(rows, "cabai_rawit_merah", REGIONS)
    assert set(df["region"]) == {"nasional", "diy"}
    assert len(df) == 3  # "-" dibuang, Jawa Tengah diabaikan
    row = df[(df["region"] == "diy") & (df["date"] == pd.Timestamp("2024-03-04"))]
    assert row["price"].iloc[0] == 61250.0
    assert (df["commodity"] == "cabai_rawit_merah").all()
