from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from src.config import FIG_DIR
from src.io_physionet import list_data_files, read_vgrf_file, side_totals
from src.preprocess import standardize_length
from src.viz import save_signal_examples


def pick_example(is_control=True) -> Path:
    files = list_data_files()
    df = files[files["is_control"] == is_control]
    # Prefer normal walking (walk == 1) if present
    df = df.sort_values(["walk", "fname"])
    if df.empty:
        raise RuntimeError("No matching files found. Check data/raw.")
    return Path(df.iloc[0]["fpath"])


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # one control & one PD example
    for is_control, tag in [(True, "control"), (False, "patient")]:
        path = pick_example(is_control=is_control)
        df = read_vgrf_file(path)
        L, R = side_totals(df)
        L, R = standardize_length(L, 12000), standardize_length(R, 12000)
        save_signal_examples(
            L,
            R,
            FIG_DIR / f"example_{tag}.png",
            title=f"Example VGRF Totals — {tag.title()} ({path.name})",
        )
        print(f"Saved {tag} example from {path.name}")


if __name__ == "__main__":
    main()
