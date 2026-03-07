"""
01_demographics.py
--------------------------------------------------------
Generates demographic summary (age, gender, counts) and plots
for the Parkinson's Mamba Time-Series Project (FAU PRL, WS25/26).

Outputs:
  - outputs/tables/demographics_summary.csv
  - outputs/tables/demographics_clean.csv
  - outputs/figures/age_distribution.box.png
  - outputs/figures/subjects_per_group.png
--------------------------------------------------------
Author: Prosenjit Chowdhury
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
META_DIR = Path("data/metadata")
RAW_DIR = Path("data/raw")
OUT_TABLES = Path("outputs/tables")
OUT_FIGURES = Path("outputs/figures")

OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIGURES.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Detect available metadata file
# ------------------------------------------------------------
meta_file = None
for ext in ["xlsx", "xls", "csv", "txt", "html"]:
    f = META_DIR / f"demographics.{ext}"
    if f.exists():
        meta_file = f
        break

if not meta_file:
    raise FileNotFoundError("No demographics file found in data/metadata/")

print(f"[INFO] Using metadata file: {meta_file.name}")

# ------------------------------------------------------------
# Load metadata
# ------------------------------------------------------------
if meta_file.suffix in [".xlsx", ".xls"]:
    meta = pd.read_excel(meta_file)
elif meta_file.suffix == ".csv":
    meta = pd.read_csv(meta_file)
elif meta_file.suffix == ".txt":
    meta = pd.read_csv(meta_file, sep="\t", engine="python")
elif meta_file.suffix == ".html":
    meta = pd.read_html(meta_file)[0]
else:
    raise ValueError("Unsupported metadata file format")

# ------------------------------------------------------------
# Identify subject ID, group, age, and sex columns
# ------------------------------------------------------------
id_col = next(
    (c for c in meta.columns if "id" in c.lower() or "subject" in c.lower()), None
)
group_col = next(
    (c for c in meta.columns if "group" in c.lower() or "diagnosis" in c.lower()), None
)
age_col = next((c for c in meta.columns if "age" in c.lower()), None)
sex_col = next(
    (c for c in meta.columns if "sex" in c.lower() or "gender" in c.lower()), None
)

if not id_col:
    raise ValueError("No subject ID column found in metadata")

meta[id_col] = meta[id_col].astype(str).str.strip()

print(
    f"[INFO] Detected columns -> ID: {id_col}, Group: {group_col}, Age: {age_col}, Sex: {sex_col}"
)

# ------------------------------------------------------------
# Match with raw signal subjects
# ------------------------------------------------------------
signal_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".txt")]
signal_subjects = pd.Series([f.split("_")[0] for f in signal_files]).unique()
meta_subjects = meta[id_col].unique()

intersection = set(signal_subjects) & set(meta_subjects)
meta_clean = meta[meta[id_col].isin(intersection)].copy()

print(f"[OK] Matched subjects retained: {len(meta_clean)} / {len(meta)}")

# ------------------------------------------------------------
# Clean & standardize columns
# ------------------------------------------------------------
if group_col:
    meta_clean[group_col] = (
        meta_clean[group_col]
        .astype(str)
        .str.replace("Control", "Control")
        .str.replace("PD", "PD")
    )

if sex_col:
    meta_clean[sex_col] = (
        meta_clean[sex_col].astype(str).str[0].str.upper()
    )  # normalize M/F

# Save cleaned metadata
meta_clean.to_csv(OUT_TABLES / "demographics_clean.csv", index=False)

# ------------------------------------------------------------
# Compute summary statistics
# ------------------------------------------------------------
summary_rows = []
for group, df in meta_clean.groupby(group_col):
    n = len(df)
    age_mean = df[age_col].mean()
    age_sd = df[age_col].std()
    males = (df[sex_col] == "M").sum()
    females = (df[sex_col] == "F").sum()
    summary_rows.append(
        {
            "Group": group,
            "Subjects": n,
            "Mean Age (±SD)": f"{age_mean:.1f} ± {age_sd:.1f}",
            "Male": males,
            "Female": females,
        }
    )

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(OUT_TABLES / "demographics_summary.csv", index=False)

print("\n[CHART] DEMOGRAPHIC SUMMARY\n", df_summary)

# ------------------------------------------------------------
# Plot 1: Boxplot of age by group
# ------------------------------------------------------------
plt.figure(figsize=(6, 4))
meta_clean.boxplot(column=age_col, by=group_col, grid=False)
plt.title("Age by Group — Boxplot")
plt.suptitle("")
plt.xlabel("Group")
plt.ylabel("Age (years)")
plt.tight_layout()
plt.savefig(OUT_FIGURES / "age_distribution.box.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# Plot 2: Subjects per group
# ------------------------------------------------------------
plt.figure(figsize=(5, 3))
df_summary.plot(
    kind="bar", x="Group", y="Subjects", color=["#1f77b4", "#ff7f0e"], legend=False
)
plt.title("Subjects per Group")
plt.ylabel("Count")
plt.xlabel("")
plt.tight_layout()
plt.savefig(OUT_FIGURES / "subjects_per_group.png", dpi=300)
plt.close()

print("[OK] Figures saved to outputs/figures/")
print("[OK] Tables saved to outputs/tables/")
