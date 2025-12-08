"""
count.py
-------------------------------------
Checks subject consistency between metadata and raw gait signal files
for the Parkinson's Mamba project.
-------------------------------------
"""

import os
import pandas as pd
from pathlib import Path

# -----------------------------
# Define paths
# -----------------------------
META_DIR = Path("data/metadata")
RAW_DIR = Path("data/raw")
OUTPUT_PATH = Path("outputs/tables/subject_mismatch_report.csv")

# -----------------------------
# Detect which demographics file exists
# -----------------------------
meta_file = None
for ext in ["xlsx", "xls", "csv", "txt", "html"]:
    candidate = META_DIR / f"demographics.{ext}"
    if candidate.exists():
        meta_file = candidate
        break

if not meta_file:
    raise FileNotFoundError("❌ No demographics file found in data/metadata/")

print(f"📂 Using metadata file: {meta_file.name}")

# -----------------------------
# Load metadata automatically
# -----------------------------
if meta_file.suffix == ".xlsx" or meta_file.suffix == ".xls":
    meta = pd.read_excel(meta_file)
elif meta_file.suffix == ".csv":
    meta = pd.read_csv(meta_file)
elif meta_file.suffix == ".txt":
    meta = pd.read_csv(meta_file, sep="\t", engine="python")
elif meta_file.suffix == ".html":
    meta = pd.read_html(meta_file)[0]
else:
    raise ValueError("❌ Unsupported file format for metadata")

print(f"✅ Metadata loaded: {meta.shape[0]} rows, {meta.shape[1]} columns")

# -----------------------------
# Identify subject column
# -----------------------------
subject_col = None
for col in meta.columns:
    if any(keyword in col.lower() for keyword in ["subject", "id", "code"]):
        subject_col = col
        break

if not subject_col:
    raise ValueError("❌ Could not find subject ID column in metadata")

print(f"🧾 Subject column detected: {subject_col}")

# -----------------------------
# Extract subject IDs
# -----------------------------
meta_subjects = meta[subject_col].astype(str).str.strip().unique()
print(f"📊 Unique subjects in metadata: {len(meta_subjects)}")

# -----------------------------
# Read signal files
# -----------------------------
signal_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".txt")]
signal_subjects = [f.split("_")[0].strip() for f in signal_files]
signal_subjects = pd.Series(signal_subjects).unique()

print(f"📈 Signal files found: {len(signal_files)}")
print(f"👣 Unique subjects from signals: {len(signal_subjects)}")

# -----------------------------
# Compare overlap
# -----------------------------
intersection = set(meta_subjects) & set(signal_subjects)
missing_in_raw = set(meta_subjects) - set(signal_subjects)
missing_in_meta = set(signal_subjects) - set(meta_subjects)

print("\n🔍 SUMMARY REPORT")
print("---------------------------")
print(f"Metadata subjects:       {len(meta_subjects)}")
print(f"Signal subjects:         {len(signal_subjects)}")
print(f"Matched (intersection):  {len(intersection)}")
print(f"Missing in raw signals:  {len(missing_in_raw)}")
print(f"Missing in metadata:     {len(missing_in_meta)}")
print("---------------------------")

# -----------------------------
# Save mismatch report safely
# -----------------------------
max_len = max(len(missing_in_raw), len(missing_in_meta))
df_mismatch = pd.DataFrame(
    {
        "missing_in_raw": list(missing_in_raw)
        + [None] * (max_len - len(missing_in_raw)),
        "missing_in_meta": list(missing_in_meta)
        + [None] * (max_len - len(missing_in_meta)),
    }
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df_mismatch.to_csv(OUTPUT_PATH, index=False)
print(f"📄 Saved detailed mismatch report to {OUTPUT_PATH.resolve()}")
