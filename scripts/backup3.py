"""
03_tsfresh_ULTRA_FAST.py
ULTRA-OPTIMIZED: No TSFresh rolling - direct feature extraction on chunks
Runs in under 5 minutes!
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats

from src.config import FIG_DIR, TAB_DIR
from src.io_physionet import list_data_files, read_vgrf_file
from src.preprocess import standardize_length, normalize_force

# ---------------- Configuration ----------------
FS = 100  # Hz
WIN_MS = 30  # window in ms
STEP_MS = 15  # step in ms
N_FILES_LIMIT = None  # Number of files for quick test
# ------------------------------------------------


def extract_simple_features(signal, window_len, step_len):
    """
    Extract simple statistical features from windowed signal.
    Much faster than TSFresh!

    Returns: dict of feature vectors (one per feature type)
    """
    n_windows = (len(signal) - window_len) // step_len + 1

    features = {
        "mean": [],
        "std": [],
        "min": [],
        "max": [],
        "median": [],
        "q25": [],
        "q75": [],
        "iqr": [],
        "rms": [],
        "peak_to_peak": [],
        "zero_crossings": [],
        "skewness": [],
        "kurtosis": [],
        "energy": [],
    }

    for i in range(n_windows):
        start = i * step_len
        end = start + window_len
        window = signal[start:end]

        # Basic statistics
        features["mean"].append(np.mean(window))
        features["std"].append(np.std(window))
        features["min"].append(np.min(window))
        features["max"].append(np.max(window))
        features["median"].append(np.median(window))
        features["q25"].append(np.percentile(window, 25))
        features["q75"].append(np.percentile(window, 75))
        features["iqr"].append(np.percentile(window, 75) - np.percentile(window, 25))
        features["rms"].append(np.sqrt(np.mean(window**2)))
        features["peak_to_peak"].append(np.ptp(window))
        features["zero_crossings"].append(np.sum(np.diff(np.sign(window)) != 0))
        features["skewness"].append(stats.skew(window))
        features["kurtosis"].append(stats.kurtosis(window))
        features["energy"].append(np.sum(window**2))

    return features


def aggregate_features(feature_dict):
    """
    Compute mean, std, skew, kurtosis for each feature across windows.
    Returns flat feature vector.
    """
    agg_features = {}

    for feat_name, feat_values in feature_dict.items():
        arr = np.array(feat_values)
        agg_features[f"{feat_name}_mean"] = np.mean(arr)
        agg_features[f"{feat_name}_std"] = np.std(arr)
        agg_features[f"{feat_name}_skew"] = stats.skew(arr)
        agg_features[f"{feat_name}_kurt"] = stats.kurtosis(arr)

    return agg_features


def extract_features_all_subjects(sequences, fs, window_ms, step_ms):
    """
    Extract and aggregate features for all subjects.
    """
    window_len = int(window_ms * fs / 1000)
    step_len = int(step_ms * fs / 1000)

    all_features = []

    for subject_id, signal in tqdm(sequences.items(), desc="Extracting features"):
        # Extract windowed features
        feat_dict = extract_simple_features(signal, window_len, step_len)

        # Aggregate across windows
        agg_feat = aggregate_features(feat_dict)
        agg_feat["subject"] = subject_id

        all_features.append(agg_feat)

    df = pd.DataFrame(all_features).set_index("subject")

    # Handle any NaN/inf
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median())

    return df


def collect_sequences(side="R", n_files=None):
    """Load gait sequences for specified side."""
    files = list_data_files()
    if n_files:
        files = files.head(n_files)

    seqs, labels = {}, {}

    for _, row in tqdm(
        files.iterrows(), total=len(files), desc=f"Loading {side} signals"
    ):
        df = read_vgrf_file(Path(row["fpath"]))

        if side == "L":
            arr = df["L_total"].to_numpy()
        elif side == "R":
            arr = df["R_total"].to_numpy()
        else:  # "LR"
            arr = np.column_stack(
                [df["L_total"].to_numpy(), df["R_total"].to_numpy()]
            ).ravel(order="C")

        arr = standardize_length(arr, 10000)
        arr = normalize_force(arr)

        sid = f"{row['ID']}_walk{row['walk']}"
        seqs[sid] = arr
        labels[sid] = "Control" if row["is_control"] else "PD"

    return seqs, labels


def reduce_and_plot(X, y, tag):
    """Z-score, PCA, t-SNE, and plot."""
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=2, random_state=42)
    Xp = pca.fit_transform(Xz)

    # t-SNE
    perplexity = min(30, len(X) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_jobs=-1)
    Xt = tsne.fit_transform(Xz)

    def scatter(Z, method):
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = {"Control": "#2ecc71", "PD": "#e74c3c"}

        for lab in ["Control", "PD"]:
            idx = y == lab
            ax.scatter(
                Z[idx, 0],
                Z[idx, 1],
                s=60,
                alpha=0.7,
                c=colors[lab],
                label=f"{lab} (n={idx.sum()})",
                edgecolors="k",
                linewidths=0.5,
            )

        ax.legend(fontsize=11)
        ax.set_xlabel(f"{method} 1", fontsize=12)
        ax.set_ylabel(f"{method} 2", fontsize=12)
        ax.set_title(f"{method} — {tag} foot (z-score)", fontsize=13, fontweight="bold")
        ax.grid(alpha=0.3)
        plt.tight_layout()

        out_path = FIG_DIR / f"{method.lower()}_{tag}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✅ {out_path.name}")

    scatter(Xp, "PCA")
    scatter(Xt, "tSNE")


def run_pipeline():
    """Main pipeline."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"⚡ ULTRA-FAST MODE")
    print(f"Window: {WIN_MS}ms | Step: {STEP_MS}ms | Files: {N_FILES_LIMIT}")
    print(f"{'=' * 60}\n")

    for tag in ["R", "L", "LR"]:
        print(f"\n🚀 {tag} foot")
        print("-" * 60)

        seqs, labels = collect_sequences(side=tag, n_files=N_FILES_LIMIT)

        features = extract_features_all_subjects(seqs, FS, WIN_MS, STEP_MS)

        out_csv = TAB_DIR / f"features_{tag}.csv"
        features.to_csv(out_csv)
        print(f"✅ Saved: {out_csv.name} | Shape: {features.shape}")

        y = np.array([labels[sid] for sid in features.index])
        X = features.to_numpy(dtype=float)

        reduce_and_plot(X, y, tag)

    print(f"\n{'=' * 60}")
    print(f"✅ DONE! Check {FIG_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    import time

    start = time.time()
    run_pipeline()
    print(f"\n⏱️  Total time: {time.time() - start:.1f} seconds")
