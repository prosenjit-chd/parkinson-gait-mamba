from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_hist_box_by_group(
    ages: pd.Series, groups: pd.Series, out_png: Path, title: str = "Age Distribution"
):
    plt.figure(figsize=(8, 4.5))
    # Histogram per group
    unique = groups.unique()
    for g in unique:
        a = ages[groups == g].dropna().astype(float)
        plt.hist(a, bins=10, alpha=0.5, label=str(g))
    plt.xlabel("Age (years)")
    plt.ylabel("Count")
    plt.title(title + " — Histogram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png.with_suffix(".hist.png"), dpi=140)
    plt.close()

    # Boxplot
    plt.figure(figsize=(6, 4))
    data = [ages[groups == g].dropna().astype(float).values for g in unique]
    plt.boxplot(data, labels=[str(g) for g in unique], vert=True)
    plt.ylabel("Age (years)")
    plt.title(title + " — Boxplot")
    plt.tight_layout()
    plt.savefig(out_png.with_suffix(".box.png"), dpi=140)
    plt.close()


def save_signal_examples(
    left: np.ndarray, right: np.ndarray, out_png: Path, title="Example VGRF (Totals)"
):
    n = min(len(left), len(right))
    t = np.arange(n) / 100.0  # 100 Hz
    plt.figure(figsize=(9, 4))
    plt.plot(t, left[:n], label="Left total")
    plt.plot(t, right[:n], label="Right total")
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()
