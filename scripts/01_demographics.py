from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from src.config import FIG_DIR, TAB_DIR, DEMOGRAPHICS_CANDIDATES
from src.viz import save_hist_box_by_group


def read_demographics() -> pd.DataFrame:
    """Try multiple formats and harmonize columns."""
    for p in DEMOGRAPHICS_CANDIDATES:
        if p.exists():
            ext = p.suffix.lower()
            if ext in [".xlsx", ".xls"]:
                df = pd.read_excel(p)
            elif ext in [".csv"]:
                df = pd.read_csv(p)
            elif ext in [".txt", ".tsv"]:
                df = pd.read_csv(p, sep=r"\s+|\t|,", engine="python")
            elif ext == ".html":
                df = pd.read_html(p)[0]
            else:
                continue
            break
    else:
        raise FileNotFoundError("No demographics file found in data/metadata/")

    # Standardize column names
    df = df.rename(columns={c: c.strip().title() for c in df.columns})
    # Make sure we have these logical columns if present:
    # ID, Group or (Co/Pt implied), Gender, Age, Height, Weight
    # If 'Group' is numeric, derive from ID string; else keep as is.
    if "Group" in df.columns and df["Group"].dtype.kind in "biufc":
        pass  # keep for now; we’ll also make a text label below
    # Derive label from ID text (Co vs Pt)
    if "Id" in df.columns:
        df["ID"] = df["Id"].astype(str)
    if "ID" not in df.columns:
        # try to find a best-guess column
        cand = [c for c in df.columns if c.lower().startswith("id")]
        if cand:
            df["ID"] = df[cand[0]].astype(str)
    df["Label"] = np.where(
        df["ID"].str.contains("Co", case=False),
        "Control",
        np.where(df["ID"].str.contains("Pt", case=False), "PD", pd.NA),
    )

    # Clean gender
    if "Gender" in df.columns:
        df["Gender"] = (
            df["Gender"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(
                {
                    "male": "Male",
                    "m": "Male",
                    "1": "Male",
                    "female": "Female",
                    "f": "Female",
                    "2": "Female",
                }
            )
            .fillna(df["Gender"])
        )
    # Numeric coercions
    for col in ["Age", "Height", "Weight"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def main():
    df = read_demographics()

    # Summary table for slides
    summary = (
        df.groupby("Label")
        .agg(
            N=("ID", "count"),
            Age_Mean=("Age", "mean"),
            Age_SD=("Age", "std"),
            Male_N=(
                "Gender",
                lambda s: (s == "Male").sum() if s.notna().any() else np.nan,
            ),
            Female_N=(
                "Gender",
                lambda s: (s == "Female").sum() if s.notna().any() else np.nan,
            ),
        )
        .reset_index()
    )
    summary["Age_Mean"] = summary["Age_Mean"].round(1)
    summary["Age_SD"] = summary["Age_SD"].round(1)

    TAB_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    summary.to_csv(TAB_DIR / "demographics_summary.csv", index=False)
    df.to_csv(TAB_DIR / "demographics_clean.csv", index=False)

    # Age distribution plots
    if "Age" in df.columns:
        save_hist_box_by_group(
            df["Age"], df["Label"], FIG_DIR / "age_distribution", title="Age by Group"
        )

    print("Saved tables ->", TAB_DIR)
    print("Saved figures ->", FIG_DIR)
    print(summary)


if __name__ == "__main__":
    main()
