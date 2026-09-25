"""Halaman 1 — Forecast: KPI, grafik historis + forecast + interval, peringatan, unduh CSV."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import (
    ACTUAL_COLOR,
    CFG,
    COMMODITIES,
    MODEL_COLORS,
    REGIONS,
    disclaimer,
    load_forecast,
    load_prices,
    model_label,
    pct,
    plot_layout,
    rupiah,
    tgl,
)

f = st.session_state["filters"]
region, commodity, model, h = f["region"], f["commodity"], f["model"], f["horizon"]
dcfg = CFG["dashboard"]

st.title(f"{COMMODITIES[commodity]} · {REGIONS[region]}")
st.caption(f"Model: **{model_label(model)}**" + (" (terbaik berdasarkan MASE backtest H7)" if f["is_best"] else "")
           + f" · Horizon {h} hari")

prices = load_prices()
hist = prices[(prices["region"] == region) & (prices["commodity"] == commodity)].sort_values("date")
fc_all = load_forecast()
fc = fc_all[(fc_all["region"] == region) & (fc_all["commodity"] == commodity) & (fc_all["model"] == model)]
fc = fc.sort_values("step")
fc_h = fc[fc["step"] <= h]

last_date, last_price = hist["date"].iloc[-1], hist["price"].iloc[-1]
pred_h = fc_h["yhat"].iloc[-1]
change = 100 * (pred_h / last_price - 1)
pred_7 = fc.loc[fc["step"] == 7, "yhat"].iloc[0]
change_7 = pred_7 / last_price - 1

# --- KPI ---------------------------------------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric(f"Harga terakhir ({tgl(last_date)})", rupiah(last_price))
c2.metric(f"Prediksi H+{h} ({tgl(fc_h['date'].iloc[-1])})", rupiah(pred_h),
          delta=pct(change), delta_color="inverse")
row_h = fc_h.iloc[-1]
c3.metric(f"Rentang 80% pada H+{h} (Rp/kg)",
          f"{rupiah(row_h['lo80'], False)[3:]}–{rupiah(row_h['hi80'], False)[3:]}",
          help="80% kemungkinan harga aktual berada di rentang ini (berdasarkan error historis model).")

# F-13: peringatan kenaikan > 10% dalam 7 hari (status warning: ikon + label, bukan warna saja)
if change_7 > dcfg["alert_threshold"]:
    st.warning(f"**Peringatan kenaikan:** harga diprediksi naik {pct(100 * change_7)} dalam 7 hari "
               f"(ambang {dcfg['alert_threshold']:.0%}). Pertimbangkan menyetok lebih awal.", icon="⚠️")
elif change_7 < -dcfg["alert_threshold"]:
    st.info(f"Harga diprediksi turun {pct(100 * change_7)} dalam 7 hari.", icon="📉")

# --- Grafik historis + forecast -------------------------------------------------------------
recent = hist[hist["date"] > last_date - pd.Timedelta(days=dcfg["history_days"])]
color = MODEL_COLORS.get(model, "#2a78d6")
rgb = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
# Sambungkan garis forecast dengan titik aktual terakhir.
anchor = pd.DataFrame({"date": [last_date], "yhat": [last_price], "lo80": [last_price], "hi80": [last_price],
                       "lo95": [last_price], "hi95": [last_price]})
band = pd.concat([anchor, fc_h[["date", "yhat", "lo80", "hi80", "lo95", "hi95"]]], ignore_index=True)

fig = go.Figure()
for lv, alpha in (("95", 0.18), ("80", 0.34)):
    fig.add_trace(go.Scatter(x=band["date"], y=band[f"hi{lv}"], mode="lines", line=dict(width=0),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=band["date"], y=band[f"lo{lv}"], mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha})",
                             name=f"Interval {lv}%", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=recent["date"], y=recent["price"], mode="lines", name="Aktual",
                         line=dict(color=ACTUAL_COLOR, width=2),
                         hovertemplate="%{y:,.0f}<extra>Aktual</extra>"))
fig.add_trace(go.Scatter(x=band["date"], y=band["yhat"], mode="lines", name=f"Forecast {model_label(model)}",
                         line=dict(color=color, width=2, dash="dot"),
                         customdata=band[["lo80", "hi80"]],
                         hovertemplate="%{y:,.0f} (80%: %{customdata[0]:,.0f}–%{customdata[1]:,.0f})"
                                       "<extra>Forecast</extra>"))
fig.add_vline(x=last_date, line=dict(color="rgba(128,128,128,0.5)", width=1, dash="dash"))
fig.add_annotation(x=last_date, y=1, yref="paper", text="hari ini", showarrow=False,
                   xanchor="right", yanchor="bottom", font=dict(size=11, color="gray"))
plot_layout(fig, height=460, yaxis_title="Rp/kg", separators=",.")
# Fokus awal: 60 hari terakhir + forecast; 180 hari historis tetap bisa digeser/di-zoom.
view_start = max(recent["date"].iloc[0], last_date - pd.Timedelta(days=max(60, 4 * h)))
fig.update_xaxes(range=[view_start, band["date"].iloc[-1] + pd.Timedelta(days=1)])
st.plotly_chart(fig, width="stretch")
st.caption(f"Menampilkan {max(60, 4 * h)} hari terakhir — geser atau zoom out untuk melihat hingga "
           f"{dcfg['history_days']} hari historis.")

# --- Tabel & unduhan ---------------------------------------------------------------------
st.subheader("Tabel forecast")
table = fc_h[["date", "yhat", "lo80", "hi80", "lo95", "hi95"]].copy()
table.insert(0, "hari_ke", range(1, len(table) + 1))
table["perubahan_%"] = 100 * (table["yhat"] / last_price - 1)
show = table.copy()
show["date"] = show["date"].map(lambda d: tgl(d, day=True))
for c in ("yhat", "lo80", "hi80", "lo95", "hi95"):
    show[c] = show[c].map(lambda v: rupiah(v, False))
show["perubahan_%"] = show["perubahan_%"].map(pct)
show.columns = ["Hari ke", "Tanggal", "Prediksi", "Bawah 80%", "Atas 80%", "Bawah 95%", "Atas 95%", "Perubahan"]
st.dataframe(show, hide_index=True, width="stretch")

csv = table.assign(region=region, commodity=commodity, model=model).round(2)
col_a, col_b = st.columns(2)
col_a.download_button("⬇️ Unduh forecast (CSV)", csv.to_csv(index=False).encode("utf-8"),
                      file_name=f"pangancast_{region}_{commodity}_{model}_h{h}.csv", mime="text/csv")
all_models = fc_all[(fc_all["region"] == region) & (fc_all["commodity"] == commodity)]
col_b.download_button("⬇️ Unduh semua model (CSV)", all_models.round(2).to_csv(index=False).encode("utf-8"),
                      file_name=f"pangancast_{region}_{commodity}_semua_model.csv", mime="text/csv")

disclaimer()
