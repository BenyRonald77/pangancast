"""PanganCast — dashboard forecasting harga pangan harian (PIHPS BI).

Jalankan: ``streamlit run app/streamlit_app.py``

Dashboard hanya membaca artefak di ``outputs/`` (hasil ``make train``), tidak
melakukan training ulang, sehingga ringan untuk Streamlit Community Cloud.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="PanganCast — Prediksi Harga Pangan", page_icon="🌶️", layout="wide")

sys.path.insert(0, str(Path(__file__).parent))
from common import artifacts_ready, sidebar_filters  # noqa: E402

if not artifacts_ready():
    st.error("Artefak model belum tersedia. Jalankan `make data` lalu `make train` terlebih dahulu.")
    st.stop()

# Sidebar dirender di entrypoint agar filter sama dan tetap tersimpan di semua halaman.
st.session_state["filters"] = sidebar_filters()

nav = st.navigation(
    [
        st.Page("pages/1_forecast.py", title="Forecast", icon="📈", default=True),
        st.Page("pages/2_perbandingan_model.py", title="Perbandingan Model", icon="🏆"),
        st.Page("pages/3_eksplorasi_data.py", title="Eksplorasi Data", icon="🔍"),
        st.Page("pages/4_tentang_proyek.py", title="Tentang Proyek", icon="ℹ️"),
    ]
)
nav.run()
