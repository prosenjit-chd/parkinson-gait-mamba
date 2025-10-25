from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from .config import DATA_RAW

# --- File naming helpers ------------------------------------------------------

ID_RE = re.compile(r"([A-Za-z]{2})(Co|Pt)(\d+)_([0-9]{2})\.txt$")  # e.g., GaCo01_01.txt


def parse_file_id(path: Path) -> Optional[Dict]:
    m = ID_RE.search(path.name)
    if not m:
        return None
    study, copt, subj, walk = m.groups()
    return {
        "ID": f"{study}{copt}{int(subj):02d}",
        "study": study,
        "is_control": (copt == "Co"),
        "is_patient": (copt == "Pt"),
        "subjnum": int(subj),
        "walk": int(walk),
        "fname": path.name,
        "fpath": str(path),
    }


def list_data_files(root: Path = DATA_RAW) -> pd.DataFrame:
    files = sorted(root.glob("*.txt"))
    rows = []
    for p in files:
        info = parse_file_id(p)
        if info:
            rows.append(info)
    return pd.DataFrame(rows)


# --- Data loading -------------------------------------------------------------

COLS = (
    ["time"]
    + [f"L{i}" for i in range(1, 9)]
    + [f"R{i}" for i in range(1, 9)]
    + ["L_total", "R_total"]
)


def read_vgrf_file(path: Path, usecols: List[int] | None = None) -> pd.DataFrame:
    """
    Read one VGRF .txt file into a DataFrame with named columns.
    The files are whitespace-delimited, sampling rate 100 Hz.
    """
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=COLS,
        engine="python",
        na_values=["NaN", "-"],
    )
    if usecols:
        df = df[[COLS[i] for i in usecols]]
    return df


def side_totals(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Return left and right total force arrays from a loaded frame."""
    return df["L_total"].to_numpy(), df["R_total"].to_numpy()
