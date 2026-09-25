"""Halaman 3 — Eksplorasi Data: tren multi-komoditas, STL, heatmap bulan × tahun, efek Lebaran."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import ACTUAL_COLOR, COMMODITIES, REGIONS, SEQ_BLUE, load_prices, plot_layout
from plotly.subplots import make_subplots

from src.calendar_id import IDULFITRI

f = st.session_state["filters"]
region, commodity = f["region"], f["commodity"]
prices = load_prices()

st.title("Eksplorasi Data")
st.caption(f"Wilayah: {REGIONS[region]} · data PIHPS pasar tradisional, "
           f"{prices['date'].min():%b %Y} – {prices['date'].max():%b %Y}")

# --- Tren multi-komoditas (small multiples: satu sumbu per panel, skala berbeda) ----------
st.subheader("Tren harga semua komoditas")
reg = prices[prices["region"] == region]
weekly = (reg.set_index("date").groupby("commodity")["price"].resample("W").mean().reset_index())
fig = make_subplots(rows=len(COMMODITIES), cols=1, shared_xaxes=True, vertical_spacing=0.03,
                    subplot_titles=list(COMMODITIES.values()))
for i, c in enumerate(COMMODITIES, start=1):
    s = weekly[weekly["commodity"] == c]
    fig.add_trace(go.Scatter(x=s["date"], y=s["price"], mode="lines", name=COMMODITIES[c],
                             line=dict(color="#2a78d6" if c == commodity else ACTUAL_COLOR,
                                       width=2 if c == commodity else 1.5),
                             hovertemplate="%{y:,.0f}<extra>" + COMMODITIES[c] + "</extra>"),
                  row=i, col=1)
plot_layout(fig, height=170 * len(COMMODITIES), showlegend=False, separators=",.")
fig.update_annotations(font_size=12, x=0, xanchor="left")
fig.update_layout(hovermode="x")
st.plotly_chart(fig, width="stretch")
st.caption("Rata-rata mingguan, Rp/kg. Tiap panel punya skala sendiri; komoditas terpilih ditandai biru.")

series = reg[reg["commodity"] == commodity].set_index("date")["price"].asfreq("D")

# --- Dekomposisi STL ----------------------------------------------------------------------
st.subheader(f"Dekomposisi musiman (STL) — {COMMODITIES[commodity]}")


@st.cache_data(show_spinner="Menghitung STL…")
def stl_weekly(values: pd.Series) -> pd.DataFrame:
    from statsmodels.tsa.seasonal import STL

    w = values.resample("W").mean()
    res = STL(w, period=52, robust=True).fit()
    return pd.DataFrame({"Observasi": w, "Tren": res.trend, "Musiman tahunan": res.seasonal,
                         "Residual": res.resid})


stl = stl_weekly(series)
fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04, subplot_titles=list(stl.columns))
for i, col in enumerate(stl.columns, start=1):
    fig.add_trace(go.Scatter(x=stl.index, y=stl[col], mode="lines", name=col,
                             line=dict(color=ACTUAL_COLOR if i == 1 else "#2a78d6", width=1.5),
                             hovertemplate="%{y:,.0f}<extra>" + col + "</extra>"), row=i, col=1)
plot_layout(fig, height=620, showlegend=False, separators=",.")
fig.update_annotations(font_size=12, x=0, xanchor="left")
fig.update_layout(hovermode="x")
st.plotly_chart(fig, width="stretch")
st.caption("STL robust pada rata-rata mingguan dengan periode 52 minggu (musiman tahunan).")

# --- Heatmap bulan × tahun ------------------------------------------------------------------
st.subheader("Harga rata-rata per bulan × tahun")
monthly = series.to_frame("price").assign(year=series.index.year, month=series.index.month)
pivot = monthly.pivot_table(index="year", columns="month", values="price", aggfunc="mean")
bulan = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
fig = go.Figure(go.Heatmap(
    z=pivot.to_numpy(), x=[bulan[m - 1] for m in pivot.columns], y=[str(y) for y in pivot.index],
    colorscale=[[i / (len(SEQ_BLUE) - 1), c] for i, c in enumerate(SEQ_BLUE)],
    xgap=2, ygap=2, colorbar=dict(title="Rp/kg"),
    hovertemplate="%{x} %{y}: Rp %{z:,.0f}/kg<extra></extra>",
))
plot_layout(fig, height=360, separators=",.")
fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)", type="category", dtick=1)
fig.update_layout(hovermode="closest")
st.plotly_chart(fig, width="stretch")

# --- Zoom periode Lebaran -------------------------------------------------------------------
st.subheader("Efek Lebaran (Idulfitri)")
rows = []
for year, d in IDULFITRI.items():
    d = pd.Timestamp(d)
    win = series.loc[d - pd.Timedelta(days=30): d + pd.Timedelta(days=21)]
    if len(win) < 45:
        continue
    base = win.iloc[:7].mean()  # indeks = 100 pada rata-rata H-30 s.d. H-24
    rows.append(pd.DataFrame({"hari": (win.index - d).days, "indeks": 100 * win / base, "tahun": year}))
if rows:
    leb = pd.concat(rows)
    years = sorted(leb["tahun"].unique())
    # Tahun = ordinal → ramp sekuensial satu hue. Langkah ekstrem dihindari agar tiap tahun
    # tetap terbaca di tema terang maupun gelap; tahun terbaru ditebalkan.
    ramp = SEQ_BLUE[1:6]
    fig = go.Figure()
    for i, y in enumerate(years):
        s = leb[leb["tahun"] == y]
        color = ramp[round(i * (len(ramp) - 1) / max(len(years) - 1, 1))]
        fig.add_trace(go.Scatter(x=s["hari"], y=s["indeks"], mode="lines", name=str(y),
                                 line=dict(color=color, width=3 if y == years[-1] else 1.3),
                                 hovertemplate="H%{x:+d}: %{y:.1f}<extra>" + str(y) + "</extra>"))
    fig.add_vline(x=0, line=dict(color="rgba(128,128,128,0.6)", width=1, dash="dash"))
    fig.add_vrect(x0=-14, x1=7, fillcolor="rgba(128,128,128,0.08)", line_width=0)
    fig.add_hline(y=100, line=dict(color="rgba(128,128,128,0.4)", width=1))
    plot_layout(fig, height=420, xaxis_title="Hari relatif terhadap Idulfitri", yaxis_title="Indeks (H-30 = 100)")
    st.plotly_chart(fig, width="stretch")
    st.caption("Area abu-abu = jendela H-14 s.d. H+7 yang dipakai sebagai fitur hari raya di model.")
