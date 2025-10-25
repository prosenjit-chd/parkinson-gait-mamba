from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from src.config import FIG_DIR, TAB_DIR
from src.io_physionet import list_data_files, read_vgrf_file
from src.preprocess import standardize_length, normalize_force
from src.features_tsfresh import tsfresh_from_sequences


# -------------------------------------------------------------
# 1. Collect sequences (L, R, or LR)
# -------------------------------------------------------------
def collect_sequences(side="R"):
    """
    Build dict of sequences for tsfresh.
    side: "L" (use L_total), "R" (R_total), or "LR" (concat L_total|R_total)
    """
    files = list_data_files()
    seqs = {}
    labels = {}

    for _, row in files.iterrows():
        path = Path(row["fpath"])
        df = read_vgrf_file(path)

        # choose side
        if side == "L":
            arr = df["L_total"].to_numpy()
        elif side == "R":
            arr = df["R_total"].to_numpy()
        else:
            arr = np.column_stack(
                [df["L_total"].to_numpy(), df["R_total"].to_numpy()]
            ).ravel()

        # standardize and normalize
        arr = standardize_length(arr, 10000)  # 100 s @ 100 Hz
        arr = normalize_force(arr)

        key = f"{row['ID']}_w{row['walk']}"
        seqs[key] = arr
        labels[key] = "Control" if row["is_control"] else "PD"

    return seqs, labels


# -------------------------------------------------------------
# 2. Safe PCA/t-SNE reduction and plotting
# -------------------------------------------------------------
def reduce_and_plot(X, y, tag):
    """Run PCA + t-SNE safely (no NaN or Inf values)."""

    # Replace inf/-inf and NaN with column medians
    X = np.where(np.isfinite(X), X, np.nan)
    col_medians = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    if col_medians.size > 0:
        X[inds] = np.take(col_medians, inds[1])
    else:
        X = np.nan_to_num(X, nan=0.0)

    # Ensure fully finite
    if not np.isfinite(X).all():
        X = np.nan_to_num(X)

    # Standardize
    Xs = StandardScaler().fit_transform(X)

    # Guard degenerate cases
    if np.all(Xs == 0):
        print(f"⚠️ All-zero or constant features for {tag}, skipping PCA/TSNE.")
        return

    # PCA
    pca = PCA(n_components=2, random_state=0)
    Xp = pca.fit_transform(Xs)

    # t-SNE
    tsne = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=30,
        random_state=0,
    )
    Xt = tsne.fit_transform(Xs)

    # Scatter plot helper
    def scatter(Z, method):
        plt.figure(figsize=(6.2, 5.2))
        for lab in ["Control", "PD"]:
            idx = np.array([lbl == lab for lbl in y])
            plt.scatter(Z[idx, 0], Z[idx, 1], s=20, label=lab)
        plt.legend()
        plt.title(f"{method} — {tag}")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"{method.lower()}_{tag}.png", dpi=140)
        plt.close()

    scatter(Xp, "PCA")
    scatter(Xt, "TSNE")


# -------------------------------------------------------------
# 3. Main function
# -------------------------------------------------------------
def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    for tag in ["R", "L", "LR"]:
        print(f"\n🚀 Extracting TSFresh features ({tag})...")
        seqs, labels = collect_sequences(side=tag)

        # Extract TSFresh features
        feats = tsfresh_from_sequences(seqs)

        # Replace NaN/Inf with medians (safety)
        feats = feats.replace([np.inf, -np.inf], np.nan)
        feats = feats.fillna(feats.median())

        # Convert to arrays
        y = np.array([labels[k] for k in feats.index])
        X = feats.to_numpy(dtype=float)

        # Replace any leftover NaN with medians again
        if np.isnan(X).any():
            X = np.nan_to_num(X, nan=np.nanmedian(X, axis=0))

        # Skip if degenerate
        if np.all(X == 0):
            print(f"⚠️ Warning: all-zero features for {tag}, skipping plot.")
            continue

        # Save cleaned features
        feats.to_csv(TAB_DIR / f"tsfresh_features_{tag}.csv")
        print(f"✅ Saved {tag} features -> {TAB_DIR / f'tsfresh_features_{tag}.csv'}")

        # Run PCA/t-SNE visualization
        reduce_and_plot(X, y, tag)

    print("\n✅ All done!")
    print("✅ Features saved in ->", TAB_DIR)
    print("✅ Figures saved in  ->", FIG_DIR)


# -------------------------------------------------------------
# Entry point
# -------------------------------------------------------------
if __name__ == "__main__":
    main()
