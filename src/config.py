from pathlib import Path

# Project roots
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_META = ROOT / "data" / "metadata"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"

# Make sure output folders exist
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# File name of demographics (we’ll auto-detect available ones too)
DEMOGRAPHICS_CANDIDATES = [
    DATA_META / "demographics.xlsx",
    DATA_META / "demographics.xls",
    DATA_META / "demographics.csv",
    DATA_META / "demographics.txt",
    DATA_META / "demographics.tsv",
    DATA_META / "demographics.html",
]
