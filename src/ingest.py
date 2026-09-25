"""F-01 — Unduh data harga PIHPS Bank Indonesia per komoditas dan wilayah.

Sumber: https://www.bi.go.id/hargapangan (Pasar Tradisional).

Endpoint ``GetGridDataKomoditas`` mengembalikan tabel lebar: satu baris per
wilayah (``Semua Provinsi`` = rata-rata nasional, lalu provinsi yang dipilih),
satu kolom per tanggal (``dd/mm/yyyy``). Respons disimpan apa adanya ke
``data/raw/`` sebagai cache; unduhan berikutnya hanya mengambil tahun berjalan.

Jalur cadangan (PRD risiko: situs berubah/memblokir): ekspor Excel manual dari
situs PIHPS diletakkan di ``data/raw/manual/`` lalu jalankan
``python -m src.ingest --from-excel <file> --commodity <slug>``.

Contoh::

    python -m src.ingest                 # semua komoditas, 2018 s.d. hari ini
    python -m src.ingest --refresh       # unduh ulang semua tahun
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from src.config import get_logger, load_config, path

log = get_logger("ingest")

_DATE_COL = re.compile(r"^\d{2}/\d{2}/\d{4}$")


def parse_price(value) -> float:
    """Ubah string harga PIHPS (``"67,900"``) menjadi float; ``"-"``/kosong → NaN."""
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value) if value > 0 else float("nan")
    s = str(value).strip().replace(",", "").replace(".", "")
    if not s or s == "-" or not s.isdigit():
        return float("nan")
    v = float(s)
    return v if v > 0 else float("nan")


def grid_to_long(rows: list[dict], commodity: str, region_names: dict[str, str]) -> pd.DataFrame:
    """Ubah baris grid PIHPS (format lebar) menjadi format panjang.

    Args:
        rows: isi ``data`` dari respons PIHPS.
        commodity: slug komoditas.
        region_names: pemetaan slug wilayah → nama baris PIHPS.

    Returns:
        DataFrame dengan kolom ``date, region, commodity, price``.
    """
    name_to_slug = {v.strip().lower(): k for k, v in region_names.items()}
    records = []
    for row in rows:
        slug = name_to_slug.get(str(row.get("name", "")).strip().lower())
        if slug is None:
            continue
        for key, val in row.items():
            if _DATE_COL.match(str(key)):
                records.append(
                    {
                        "date": pd.to_datetime(key, format="%d/%m/%Y"),
                        "region": slug,
                        "commodity": commodity,
                        "price": parse_price(val),
                    }
                )
    df = pd.DataFrame(records, columns=["date", "region", "commodity", "price"])
    return df.dropna(subset=["price"])


class PIHPSClient:
    """Klien sederhana untuk endpoint tabel harga PIHPS dengan jeda dan retry."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (PanganCast research; +https://github.com/)",
                "Referer": cfg["referer"],
                "X-Requested-With": "XMLHttpRequest",
            }
        )
        self._last = 0.0

    def _throttle(self) -> None:
        wait = self.cfg["request_delay_s"] - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.time()

    def get_grid(self, commodity_id: str, province_id: int | None, start: str, end: str) -> dict:
        """Ambil grid harga satu komoditas untuk rentang tanggal tertentu."""
        params = {
            "price_type_id": self.cfg["price_type_id"],
            "comcat_id": commodity_id,
            "province_id": "" if province_id is None else province_id,
            "regency_id": "",
            "showKota": "false",
            "showPasar": "false",
            "tipe_laporan": 1,  # harian
            "start_date": start,
            "end_date": end,
        }
        url = f"{self.cfg['base_url']}/GetGridDataKomoditas"
        for attempt in range(1, self.cfg["max_retries"] + 1):
            self._throttle()
            try:
                r = self.session.get(url, params=params, timeout=self.cfg["timeout_s"])
                r.raise_for_status()
                return r.json()
            except (requests.RequestException, ValueError) as e:
                log.warning("Gagal (%s/%s) %s %s–%s: %s", attempt, self.cfg["max_retries"],
                            commodity_id, start, end, e)
                time.sleep(5 * attempt)
        raise RuntimeError(f"PIHPS tidak merespons untuk {commodity_id} {start}–{end}")


def _province_ids(cfg: dict) -> str | None:
    ids = [str(r["province_id"]) for r in cfg["regions"].values() if r["province_id"] is not None]
    return ",".join(ids) or None


def download(refresh: bool = False, commodities: list[str] | None = None) -> list[Path]:
    """Unduh data per komoditas per tahun ke ``data/raw/`` (dengan cache lokal).

    Satu request per komoditas-tahun sudah memuat baris Nasional dan provinsi
    yang dipilih, sehingga beban ke server BI minimal.
    """
    cfg = load_config()
    pcfg = cfg["pihps"]
    raw_dir = path("raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    client = PIHPSClient(pcfg)

    start = date.fromisoformat(pcfg["start_date"])
    end = date.fromisoformat(pcfg["end_date"]) if pcfg["end_date"] else date.today()
    province = _province_ids(cfg)
    files = []
    for slug, meta in cfg["commodities"].items():
        if commodities and slug not in commodities:
            continue
        for year in range(start.year, end.year + 1):
            out = raw_dir / f"pihps_{slug}_{year}.json"
            is_current = year == end.year
            if out.exists() and not refresh and not is_current:
                files.append(out)
                continue
            y_start = max(start, date(year, 1, 1)).isoformat()
            y_end = min(end, date(year, 12, 31)).isoformat()
            log.info("Unduh %s %s (%s s.d. %s)", slug, year, y_start, y_end)
            payload = client.get_grid(meta["pihps_id"], province, y_start, y_end)
            payload["_meta"] = {
                "commodity": slug,
                "pihps_id": meta["pihps_id"],
                "province_id": province,
                "start_date": y_start,
                "end_date": y_end,
                "downloaded_at": pd.Timestamp.now().isoformat(timespec="seconds"),
                "source": "PIHPS Nasional - Bank Indonesia (bi.go.id/hargapangan)",
            }
            out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            files.append(out)
    return files


def load_raw(raw_dir: Path | None = None) -> pd.DataFrame:
    """Gabungkan semua file mentah (JSON API dan Excel manual) menjadi format panjang."""
    cfg = load_config()
    raw_dir = raw_dir or path("raw")
    region_names = {k: v["pihps_name"] for k, v in cfg["regions"].items()}
    frames = []
    for f in sorted(raw_dir.glob("pihps_*.json")):
        payload = json.loads(f.read_text(encoding="utf-8"))
        slug = payload.get("_meta", {}).get("commodity") or f.stem.split("_", 1)[1].rsplit("_", 1)[0]
        frames.append(grid_to_long(payload.get("data", []), slug, region_names))
    manual = raw_dir / "manual"
    if manual.exists():
        for f in sorted(manual.glob("*.parquet")):
            frames.append(pd.read_parquet(f))
    if not frames:
        raise FileNotFoundError(f"Tidak ada data mentah di {raw_dir}. Jalankan `make data`.")
    df = pd.concat(frames, ignore_index=True)
    # Jika tanggal tumpang tindih (unduhan ulang), ambil yang terakhir.
    return df.drop_duplicates(["date", "region", "commodity"], keep="last").sort_values(
        ["commodity", "region", "date"]
    ).reset_index(drop=True)


def import_excel(xlsx: Path, commodity: str) -> Path:
    """Jalur manual: konversi ekspor Excel tabel PIHPS ke format panjang.

    Excel hasil tombol *Download* di situs PIHPS berformat lebar (kolom
    ``Komoditas (Rp)``/``name`` + kolom tanggal ``dd/mm/yyyy``).
    """
    cfg = load_config()
    region_names = {k: v["pihps_name"] for k, v in cfg["regions"].items()}
    wide = pd.read_excel(xlsx, dtype=str)
    name_col = next(c for c in wide.columns if not _DATE_COL.match(str(c)) and c.lower() not in ("no",))
    rows = wide.rename(columns={name_col: "name"}).to_dict("records")
    df = grid_to_long(rows, commodity, region_names)
    out_dir = path("raw") / "manual"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{commodity}_{xlsx.stem}.parquet"
    df.to_parquet(out, index=False)
    log.info("Impor %d baris dari %s → %s", len(df), xlsx, out)
    return out


def main() -> None:
    """Entry point CLI."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="unduh ulang semua tahun (abaikan cache)")
    ap.add_argument("--commodity", nargs="*", help="slug komoditas (default: semua)")
    ap.add_argument("--from-excel", type=Path, help="impor ekspor Excel manual PIHPS")
    args = ap.parse_args()

    if args.from_excel:
        if not args.commodity or len(args.commodity) != 1:
            ap.error("--from-excel membutuhkan tepat satu --commodity")
        import_excel(args.from_excel, args.commodity[0])
        return
    files = download(refresh=args.refresh, commodities=args.commodity)
    df = load_raw()
    log.info("Selesai: %d file mentah, %d observasi, %s s.d. %s",
             len(files), len(df), df["date"].min().date(), df["date"].max().date())


if __name__ == "__main__":
    main()
