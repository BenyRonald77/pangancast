"""Halaman 2 — Perbandingan Model: leaderboard, error per horizon, overlay holdout."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import (
    ACTUAL_COLOR,
    BASELINES,
    COMMODITIES,
    MODEL_COLORS,
    REGIONS,
    load_best,
    load_dm,
    load_metrics,
    load_predictions,
    model_label,
    plot_layout,
)

f = st.session_state["filters"]
region, commodity, h = f["region"], f["commodity"], f["horizon"]

st.title("Perbandingan Model")
st.caption(f"{COMMODITIES[commodity]} · {REGIONS[region]} — semua model memakai fold, horizon, dan data yang sama.")

metrics = load_metrics()
split = st.segmented_control(
    "Periode evaluasi", ["backtest", "holdout"], default="backtest",
    format_func={"backtest": "Backtest (12 fold rolling-origin)", "holdout": "Holdout 90 hari terakhir"}.get,
) or "backtest"

m = metrics[(metrics["region"] == region) & (metrics["commodity"] == commodity)
            & (metrics["split"] == split) & (metrics["horizon"] == h)].sort_values("mase")
winner = f["best"]  # pemenang resmi: model kandidat dengan MASE backtest H7 terendah

# --- Leaderboard ---------------------------------------------------------------------------
st.subheader(f"Leaderboard — horizon {h} hari")
lb = pd.DataFrame({
    "Peringkat": range(1, len(m) + 1),
    "Model": [("🏆 " if mm == winner else "") + model_label(mm) + (" (baseline)" if mm in BASELINES else "")
              for mm in m["model"]],
    "MASE": m["mase"].to_numpy(),
    "MASE vs naive": m["rel_mase"].to_numpy(),
    "MAPE (%)": m["mape"].to_numpy(),
    "MAE (Rp/kg)": m["mae"].to_numpy(),
    "RMSE (Rp/kg)": m["rmse"].to_numpy(),
    "Coverage 80%": m["coverage80"].to_numpy(),
    "Coverage 95%": m["coverage95"].to_numpy(),
    "Arah benar (%)": m["directional_acc"].to_numpy(),
    "Train/fold (s)": m["train_time_s"].to_numpy(),
})
num = lambda d: (lambda v: "–" if pd.isna(v) else f"{v:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", "."))  # noqa: E731
st.dataframe(
    lb.style.apply(lambda r: ["font-weight: 700" if "🏆" in r["Model"] else "" for _ in r], axis=1)
    .format({"MASE": num(3), "MASE vs naive": num(3), "MAPE (%)": num(2), "MAE (Rp/kg)": num(0),
             "RMSE (Rp/kg)": num(0), "Coverage 80%": num(1), "Coverage 95%": num(1), "Arah benar (%)": num(1),
             "Train/fold (s)": num(2)}),
    hide_index=True, width="stretch",
)
st.caption("🏆 = model terbaik resmi (dari 4 model kandidat, MASE backtest horizon 7 hari). "
           "**MASE vs naive** < 1 berarti mengalahkan baseline naive (harga kemarin) pada fold yang sama. "
           "Coverage ideal ≈ 80% dan 95%.")

# --- Error per langkah ---------------------------------------------------------------------
st.subheader("Error menurut jarak prediksi")
preds = load_predictions()
p = preds[(preds["region"] == region) & (preds["commodity"] == commodity) & (preds["split"] == split)]
p = p.assign(ape=100 * (p["y_true"] - p["yhat"]).abs() / p["y_true"])
per_step = p.groupby(["model", "step"])["ape"].mean().reset_index()

fig = go.Figure()
level = per_step.groupby("model")["ape"].mean()
hidden = []
for mm in m["model"]:  # urutan legenda = peringkat, warna = identitas model
    s = per_step[per_step["model"] == mm]
    base = mm in BASELINES
    # Model dengan error jauh di atas yang lain disembunyikan (klik legenda untuk menampilkan)
    # agar skala sumbu tidak memampatkan perbedaan antar model yang bersaing.
    far = level[mm] > 2.5 * level.min()
    if far:
        hidden.append(model_label(mm))
    fig.add_trace(go.Scatter(
        x=s["step"], y=s["ape"], mode="lines", name=model_label(mm),
        line=dict(color=MODEL_COLORS[mm], width=2, dash="dash" if base else "solid"),
        visible="legendonly" if far else True,
        hovertemplate="%{y:.2f}%<extra>" + model_label(mm) + "</extra>",
    ))
for hh in (7, 14, 30):
    fig.add_vline(x=hh, line=dict(color="rgba(128,128,128,0.35)", width=1, dash="dot"))
plot_layout(fig, height=380, xaxis_title="Hari ke depan", yaxis_title="MAPE (%)")
st.plotly_chart(fig, width="stretch")
if hidden:
    st.caption(f"Disembunyikan karena error-nya jauh lebih besar: {', '.join(hidden)} — "
               "klik legenda untuk menampilkan.")

# --- Overlay holdout -------------------------------------------------------------------------
st.subheader("Prediksi semua model pada periode holdout")
hold = preds[(preds["region"] == region) & (preds["commodity"] == commodity) & (preds["split"] == "holdout")]
chosen = st.multiselect("Model yang ditampilkan", list(m["model"]),
                        default=[mm for mm in m["model"] if mm not in BASELINES and model_label(mm) not in hidden]
                        or list(m["model"])[:2],
                        format_func=model_label)
fig2 = go.Figure()
actual = hold.drop_duplicates("date").sort_values("date")
fig2.add_trace(go.Scatter(x=actual["date"], y=actual["y_true"], mode="lines", name="Aktual",
                          line=dict(color=ACTUAL_COLOR, width=2.5),
                          hovertemplate="%{y:,.0f}<extra>Aktual</extra>"))
for mm in chosen:
    s = hold[hold["model"] == mm].sort_values("date")
    # Putus garis antar-origin agar tiap forecast 30 hari terlihat terpisah.
    xs, ys = [], []
    for _, g in s.groupby("fold"):
        xs += list(g["date"]) + [None]
        ys += list(g["yhat"]) + [None]
    fig2.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=model_label(mm),
                              line=dict(color=MODEL_COLORS[mm], width=2, dash="dash" if mm in BASELINES else "solid"),
                              hovertemplate="%{y:,.0f}<extra>" + model_label(mm) + "</extra>"))
for o in hold["origin"].drop_duplicates():
    fig2.add_vline(x=o, line=dict(color="rgba(128,128,128,0.35)", width=1, dash="dot"))
plot_layout(fig2, height=420, yaxis_title="Rp/kg", separators=",.")
st.plotly_chart(fig2, width="stretch")
st.caption("Garis putus-putus vertikal = origin forecast. Setiap origin memprediksi 30 hari ke depan "
           "tanpa melihat data sesudahnya.")

# --- Ringkasan semua deret -----------------------------------------------------------------
st.subheader("Model terbaik per komoditas × wilayah (MASE backtest H7)")
best = load_best().copy()
hold7 = metrics[(metrics["split"] == "holdout") & (metrics["horizon"] == 7)]
best = best.merge(hold7[["region", "commodity", "model", "mape", "rel_mase"]],
                  on=["region", "commodity", "model"], how="left", suffixes=("", "_holdout"))
summary = pd.DataFrame({
    "Komoditas": best["commodity"].map(COMMODITIES),
    "Wilayah": best["region"].map(REGIONS),
    "Model terbaik": best["model"].map(model_label),
    "MASE": best["mase"].map(num(3)),
    "MASE vs naive": best["rel_mase"].map(num(3)),
    "MAPE backtest (%)": best["mape"].map(num(2)),
    "MAPE holdout (%)": best["mape_holdout"].map(num(2)),
    "Kalahkan naive?": np.select([best["rel_mase"].isna(), best["rel_mase"] < 1],
                                 ["– harga konstan", "✅ Ya"], "❌ Tidak"),
})
st.dataframe(summary, hide_index=True, width="stretch")

dm = load_dm()
if len(dm):
    with st.expander("Uji signifikansi Diebold-Mariano (pemenang vs runner-up, H7)"):
        d = dm.copy()
        d["Komoditas"] = d["commodity"].map(COMMODITIES)
        d["Wilayah"] = d["region"].map(REGIONS)
        d["Pemenang"] = d["best"].map(model_label)
        d["Runner-up"] = d["runner_up"].map(model_label)
        d["Signifikan (5%)"] = np.where(d["significant_5pct"], "Ya", "Tidak")
        st.dataframe(d[["Komoditas", "Wilayah", "Pemenang", "Runner-up", "dm_stat", "p_value", "Signifikan (5%)"]]
                     .rename(columns={"dm_stat": "Statistik DM", "p_value": "p-value"}).round(4),
                     hide_index=True, width="stretch")
        st.caption("H0: akurasi kedua model sama (loss kuadrat, koreksi Harvey-Leybourne-Newbold). "
                   "Statistik negatif = pemenang lebih akurat. p ≥ 0,05 berarti selisihnya belum signifikan.")
