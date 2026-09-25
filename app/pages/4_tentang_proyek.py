"""Halaman 4 — Tentang Proyek: metodologi ringkas, sumber data, keterbatasan."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from common import CFG, COMMODITIES, REGIONS, load_best, load_run_info, model_label

info = load_run_info()
st.title("Tentang PanganCast")

st.markdown(
    """
**PanganCast** memprediksi harga harian cabai, bawang, dan beras di pasar tradisional
untuk 7, 14, dan 30 hari ke depan, memakai data **PIHPS Nasional Bank Indonesia**.
Empat model — SARIMA, Prophet, XGBoost, LSTM — dibandingkan secara adil dengan baseline naive
memakai protokol validasi yang sama persis.
"""
)

c1, c2, c3 = st.columns(3)
c1.metric("Deret waktu", f"{len(COMMODITIES)} komoditas × {len(REGIONS)} wilayah")
if info:
    c2.metric("Periode data", f"{pd.Timestamp(info['data_start']):%b %Y} – {pd.Timestamp(info['data_end']):%d %b %Y}")
    c3.metric("Terakhir dilatih", f"{pd.Timestamp(info['run_at']):%d %b %Y}")

st.subheader("Metodologi")
b = CFG["backtest"]
st.markdown(
    f"""
1. **Data** — diunduh dari API tabel harga PIHPS (pasar tradisional), wilayah Nasional dan
   DI Yogyakarta, sejak Januari 2018. PIHPS hanya mencatat hari kerja; akhir pekan/libur ≤ 3 hari
   diisi *forward-fill*, celah lebih panjang diinterpolasi, dan semuanya ditandai `is_imputed`.
   Salah input yang jelas (lonjakan satu hari yang langsung kembali, mis. laporan satu pasar saat
   libur) dibuang; lonjakan nyata hari raya dipertahankan.
2. **Transformasi** — semua model dilatih pada *log harga* untuk menstabilkan varians.
3. **Model**
   - *Baseline*: naive (harga besok = hari ini) dan seasonal naive (lag 7).
   - *SARIMA*: orde dipilih `auto_arima` (uji ADF, musiman mingguan m=7).
   - *Prophet*: musiman mingguan + tahunan, regresor Idulfitri, Iduladha, Natal, Tahun Baru (H-14 s.d. H+7).
   - *XGBoost*: strategi *direct* — satu model per horizon (7, 14, 30) dengan fitur lag 1–30 hari,
     rolling mean/std 7/14/30, kalender, dan hari raya.
   - *LSTM* (PyTorch): jendela input, output vektor 30 langkah, dropout + early stopping,
     scaler Min-Max hanya dari data train.
4. **Validasi** — *rolling-origin* (expanding window): {b['n_folds']} fold × {b['test_size']} hari,
   lalu **holdout {b['holdout_days']} hari terakhir** yang tidak disentuh saat tuning.
   Tuning (Optuna) hanya memakai fold backtest pada dua komoditas representatif.
5. **Metrik** — MAE, RMSE, MAPE, **MASE** (metrik utama), coverage interval 80%/95%, dan akurasi arah.
   Pemenang per deret = MASE backtest terendah pada horizon 7 hari.
6. **Interval prediksi** — SARIMA & naive: analitik; Prophet: kuantil *predictive samples*;
   XGBoost & LSTM: kuantil residual out-of-sample (*split conformal*).
"""
)

st.subheader("Model terbaik saat ini")
best = load_best()
st.dataframe(
    pd.DataFrame({
        "Komoditas": best["commodity"].map(COMMODITIES),
        "Wilayah": best["region"].map(REGIONS),
        "Model": best["model"].map(model_label),
        "MAPE H7 (%)": best["mape"].round(2),
    }),
    hide_index=True, width="stretch",
)

st.subheader("Keterbatasan")
st.markdown(
    """
- Prediksi hanya berbasis pola historis harga dan kalender; **cuaca, kebijakan impor, dan kurs belum
  dimasukkan**, sehingga lonjakan akibat *shock* (gagal panen, banjir) sulit diantisipasi.
- Rata-rata nasional dan provinsi menyembunyikan variasi antarpasar; harga di pasar Anda bisa berbeda.
- Nilai pada akhir pekan/libur adalah hasil imputasi, bukan observasi.
- Interval 80%/95% adalah perkiraan statistik; lihat kolom *coverage* di halaman Perbandingan Model
  untuk seberapa sering interval tersebut benar-benar memuat harga aktual.
- Dashboard menampilkan forecast yang dihitung saat training terakhir, bukan real-time.
"""
)

st.subheader("Sumber & lisensi")
st.markdown(
    f"""
- Data: [PIHPS Nasional — Bank Indonesia](https://www.bi.go.id/hargapangan), jenis pasar tradisional.
  Hak atas data ada pada Bank Indonesia; proyek ini hanya mengolahnya untuk tujuan riset/edukasi.
- Kode: lisensi MIT — [{CFG['dashboard']['github_url']}]({CFG['dashboard']['github_url']})
"""
)
st.caption("Prediksi bersifat indikatif, bukan saran transaksi.")
