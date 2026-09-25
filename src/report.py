"""Perbarui tabel hasil di README dari ``outputs/metrics.csv`` (sumber tunggal angka).

Bagian README di antara penanda ``<!-- RESULTS:START -->`` dan
``<!-- RESULTS:END -->`` ditulis ulang otomatis. Jalankan: ``python -m src.report``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import ROOT, load_config, path
from src.models import MODEL_LABELS

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def _fmt(x: float, d: int = 2) -> str:
    return "–" if pd.isna(x) else f"{x:.{d}f}"


def results_markdown() -> str:
    """Susun markdown ringkasan hasil dari artefak ``outputs/``."""
    cfg = load_config()
    out = path("outputs")
    metrics = pd.read_csv(out / "metrics.csv")
    best = pd.read_csv(out / "best_models.csv")
    info = json.loads((out / "run_info.json").read_text(encoding="utf-8"))
    com = {k: v["label"] for k, v in cfg["commodities"].items()}
    reg = {k: v["label"] for k, v in cfg["regions"].items()}

    hold = metrics[(metrics["split"] == "holdout") & (metrics["horizon"] == 7)]
    best = best.merge(hold[["region", "commodity", "model", "mape", "rel_mase"]],
                      on=["region", "commodity", "model"], how="left", suffixes=("", "_hold"))
    lines = [
        f"_Data PIHPS {info['data_start']} s.d. {info['data_end']}, dilatih {info['run_at'][:10]}. "
        f"Backtest {info['folds']['n_folds']} fold × {info['folds']['test_size']} hari; "
        f"holdout {info['folds']['holdout_days']} hari terakhir._",
        "",
        "**Model terbaik per deret (MASE backtest, horizon 7 hari)**",
        "",
        "| Komoditas | Wilayah | Model terbaik | MAPE backtest | MAPE holdout | MASE vs naive | Target MAPE | Lolos? |",
        "|---|---|---|---:|---:|---:|---:|:---:|",
    ]
    for _, r in best.sort_values(["commodity", "region"]).iterrows():
        target = 3.0 if r["commodity"].startswith("beras") else 12.0
        if pd.isna(r["rel_mase"]):
            ok = "–"  # harga konstan selama test: naive sempurna, rasio tak terdefinisi
        elif r["mape"] < target:
            ok = "✅" if r["rel_mase"] < 1 else "⚠️"
        else:
            ok = "❌"
        lines.append(
            f"| {com[r['commodity']]} | {reg[r['region']]} | {MODEL_LABELS[r['model']]} | {_fmt(r['mape'])}% | "
            f"{_fmt(r['mape_hold'])}% | {_fmt(r['rel_mase'], 3)} | < {target:.0f}% | {ok} |"
        )
    lines += [
        "",
        "✅ = MAPE di bawah target **dan** mengalahkan naive · ⚠️ = MAPE di bawah target tetapi tidak "
        "mengalahkan naive · ❌ = MAPE di atas target · – = harga konstan selama periode uji "
        "(naive sempurna, MASE relatif tidak terdefinisi).",
        "",
        "**Semua deret (backtest) per model dan horizon** — MAPE rata-rata (%) / MASE relatif terhadap naive "
        "(rata-rata geometrik) / jumlah deret yang mengalahkan naive",
        "",
    ]
    bt = metrics[metrics["split"] == "backtest"]
    n_series = bt[["region", "commodity"]].drop_duplicates().shape[0]

    def _gmean(x: pd.Series) -> float:
        x = x.dropna()
        return float(np.exp(np.log(x).mean())) if len(x) else float("nan")

    agg = bt.groupby(["model", "horizon"]).agg(
        mape=("mape", "mean"), rel=("rel_mase", _gmean), wins=("rel_mase", lambda x: int((x < 1).sum())))
    horizons = sorted(metrics["horizon"].unique())
    lines.append("| Model | " + " | ".join(f"H{h}" for h in horizons) + " | Waktu train/fold |")
    lines.append("|---|" + "---:|" * (len(horizons) + 1))
    times = metrics.drop_duplicates(["region", "commodity", "model"]).groupby("model")["train_time_s"].mean()
    order = agg.xs(7, level="horizon")["rel"].sort_values().index
    for m in order:
        cells = []
        for h in horizons:
            a = agg.loc[(m, h)]
            wins = "–" if m == "naive" else f"{int(a['wins'])}/{n_series}"
            cells.append(f"{_fmt(a['mape'])} / {_fmt(a['rel'], 3)} / {wins}")
        lines.append(f"| {MODEL_LABELS[m]} | " + " | ".join(cells) + f" | {_fmt(times[m], 2)} s |")
    cov = metrics[(metrics["split"] == "backtest") & (metrics["horizon"] == 7)].groupby("model")[
        ["coverage80", "coverage95"]].mean()
    lines += ["", "**Coverage interval prediksi (backtest H7, ideal 80% / 95%)**", "",
              "| Model | Coverage 80% | Coverage 95% |", "|---|---:|---:|"]
    for m in order:
        lines.append(f"| {MODEL_LABELS[m]} | {_fmt(cov.loc[m, 'coverage80'], 1)}% | "
                     f"{_fmt(cov.loc[m, 'coverage95'], 1)}% |")
    return "\n".join(lines)


def update_readme() -> None:
    """Tulis ulang bagian hasil di README.md."""
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise ValueError("Penanda RESULTS tidak ditemukan di README.md")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    readme.write_text(f"{head}{START}\n{results_markdown()}\n{END}{tail}", encoding="utf-8")


def main() -> None:
    """Entry point CLI."""
    update_readme()
    print("README.md diperbarui dari outputs/metrics.csv")


if __name__ == "__main__":
    main()
