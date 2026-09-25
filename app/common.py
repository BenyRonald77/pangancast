"""Utilitas bersama dashboard: pemuatan artefak (cache), filter sidebar, format, warna."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.models import BASELINES, MODEL_LABELS  # noqa: E402

OUT = ROOT / "outputs"
CFG = load_config()
COMMODITIES = {k: v["label"] for k, v in CFG["commodities"].items()}
REGIONS = {k: v["label"] for k, v in CFG["regions"].items()}
HORIZONS = CFG["backtest"]["horizons"]
BEST = "__best__"

# Warna mengikuti entitas model (urutan slot tetap, tidak pernah berdasarkan peringkat).
# Palet kategorikal tervalidasi (skill dataviz, palette.md); baseline memakai abu-abu.
MODEL_COLORS = {
    "arima": "#2a78d6",
    "prophet": "#eb6834",
    "xgboost": "#1baf7a",
    "lstm": "#7466d9",  # violet langkah tengah: terbaca di tema terang & gelap
    "naive": "#8a8983",
    "seasonal_naive": "#b5b4ad",
}
ACTUAL_COLOR = "#77766f"  # abu-abu tengah: kontras cukup di kedua tema
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def rupiah(x: float, unit: bool = True) -> str:
    """Format Rupiah Indonesia: ``Rp 45.250/kg``."""
    if pd.isna(x):
        return "–"
    s = f"{x:,.0f}".replace(",", ".")
    return f"Rp {s}/kg" if unit else f"Rp {s}"


def pct(x: float, digits: int = 1, sign: bool = True) -> str:
    """Format persen gaya Indonesia (koma desimal)."""
    if pd.isna(x):
        return "–"
    s = f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}"
    return s.replace(".", ",") + "%"


HARI = ["Sen", "Sel", "Rab", "Kam", "Jum", "Sab", "Min"]
BULAN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]


def tgl(d, day: bool = False, year: bool = True) -> str:
    """Tanggal gaya Indonesia, mis. ``Kam, 25 Sep 2026``."""
    d = pd.Timestamp(d)
    s = f"{d.day:02d} {BULAN[d.month - 1]}" + (f" {d.year}" if year else "")
    return f"{HARI[d.dayofweek]}, {s}" if day else s


def model_label(m: str) -> str:
    return MODEL_LABELS.get(m, m)


@st.cache_data(show_spinner=False)
def load_prices() -> pd.DataFrame:
    return pd.read_parquet(ROOT / CFG["paths"]["processed"])


@st.cache_data(show_spinner=False)
def load_forecast() -> pd.DataFrame:
    return pd.read_parquet(OUT / "forecast.parquet")


@st.cache_data(show_spinner=False)
def load_metrics() -> pd.DataFrame:
    return pd.read_csv(OUT / "metrics.csv")


@st.cache_data(show_spinner=False)
def load_best() -> pd.DataFrame:
    return pd.read_csv(OUT / "best_models.csv")


@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    return pd.read_parquet(OUT / "backtest_predictions.parquet")


@st.cache_data(show_spinner=False)
def load_dm() -> pd.DataFrame:
    f = OUT / "dm_test.csv"
    return pd.read_csv(f) if f.exists() and f.stat().st_size > 1 else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_run_info() -> dict:
    f = OUT / "run_info.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def artifacts_ready() -> bool:
    return all((OUT / f).exists() for f in ("forecast.parquet", "metrics.csv", "best_models.csv"))


def best_model_for(region: str, commodity: str) -> str:
    b = load_best()
    row = b[(b["region"] == region) & (b["commodity"] == commodity)]
    return row["model"].iloc[0] if len(row) else "naive"


def _keep(key: str) -> None:
    """Salin nilai widget ke key bayangan agar tidak hilang saat Streamlit membuang state widget."""
    st.session_state[f"_{key}"] = st.session_state[key]


def _persistent(widget, label: str, options: list, key: str, **kw):
    """Widget yang nilainya bertahan lintas halaman (dipulihkan dari key bayangan ``_key``)."""
    ss = st.session_state
    value = ss.get(key, ss[f"_{key}"])
    if value not in options:
        value = options[0]
    ss[f"_{key}"] = value
    if key not in ss:  # widget state dibuang → pulihkan dari key bayangan
        ss[key] = value
    return widget(label, options, key=key, on_change=_keep, args=(key,), **kw)


def sidebar_filters() -> dict:
    """Filter sidebar yang sama untuk semua halaman (state dipertahankan antar halaman)."""
    d = CFG["dashboard"]
    ss = st.session_state
    ss.setdefault("_f_commodity", d["default_commodity"])
    ss.setdefault("_f_region", d["default_region"])
    ss.setdefault("_f_model", BEST)
    ss.setdefault("_f_horizon", d["default_horizon"])

    with st.sidebar:
        st.header("Filter")
        commodity = _persistent(st.selectbox, "Komoditas", list(COMMODITIES), "f_commodity",
                                format_func=COMMODITIES.get)
        region = _persistent(st.selectbox, "Wilayah", list(REGIONS), "f_region", format_func=REGIONS.get)
        best = best_model_for(region, commodity)
        available = set(load_forecast()["model"])
        models = [m for m in MODEL_LABELS if m in available]
        choice = _persistent(
            st.selectbox, "Model", [BEST, *models], "f_model",
            format_func=lambda m: f"Terbaik ({model_label(best)})" if m == BEST else model_label(m),
        )
        horizon = _persistent(st.radio, "Horizon", HORIZONS, "f_horizon", horizontal=True,
                              format_func=lambda h: f"{h} hari")
        info = load_run_info()
        if info:
            st.caption(f"Data PIHPS s.d. **{pd.Timestamp(info['data_end']):%d %b %Y}**  \n"
                       f"Model dilatih {pd.Timestamp(info['run_at']):%d %b %Y}")
    model = best if choice == BEST else choice
    return {"commodity": commodity, "region": region, "model": model,
            "is_best": model == best, "best": best, "horizon": int(horizon)}


def disclaimer() -> None:
    st.caption("⚠️ Prediksi bersifat indikatif berdasarkan pola historis, **bukan saran transaksi**. "
               "Sumber data: PIHPS Nasional, Bank Indonesia (bi.go.id/hargapangan), pasar tradisional.")


def plot_layout(fig, height: int = 420, **kw):
    """Tata letak Plotly yang ringan: grid samar, legenda di atas, hover terpadu."""
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified", **kw,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    return fig


__all__ = ["BASELINES"]
