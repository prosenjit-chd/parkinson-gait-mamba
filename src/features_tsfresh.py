from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from tsfresh.feature_extraction import extract_features, EfficientFCParameters

# ---------- helpers ----------


def _ms_to_samples(ms: float, fs: int) -> int:
    # round to nearest >= 1
    return max(1, int(round(ms * fs / 1000.0)))


def _rolling_windows(x: np.ndarray, win: int, step: int) -> np.ndarray:
    """Return (n_windows, win) using a simple Python loop (safe on Windows)."""
    if len(x) < win:
        return np.empty((0, win))
    out = []
    i = 0
    while i + win <= len(x):
        out.append(x[i : i + win])
        i += step
    return np.stack(out, axis=0) if out else np.empty((0, win))


# ---------- main APIs ----------


def tsfresh_from_sequences_windowed(
    seqs: Dict[str, np.ndarray],
    fs: int = 100,
    window_ms: int = 30,
    step_ms: int = 15,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract TSFresh features on sliding windows for every subject sequence,
    then aggregate per subject with mean/std/skew/kurt of each feature.
    Returns (aggregated_features_df, subject_index_series).

    seqs: dict {subject_id: 1D array}
    """
    win = _ms_to_samples(window_ms, fs)  # e.g., 30ms @100Hz -> 3 samples
    step = _ms_to_samples(step_ms, fs)  # e.g., 15ms @100Hz -> 2 samples

    # ---- build long dataframe with one "id" per (subject, window) ----
    long_records = []
    subj_for_window = []  # map window-id -> subject
    for sid, arr in seqs.items():
        arr = np.asarray(arr, dtype=float)
        wins = _rolling_windows(arr, win=win, step=step)  # shape (W, win)
        for wi, w in enumerate(wins):
            # id = unique per window
            wid = f"{sid}__w{wi:05d}"
            subj_for_window.append((wid, sid))
            long_records.append(
                pd.DataFrame({"id": wid, "time": np.arange(win, dtype=int), "value": w})
            )
    if not long_records:
        return pd.DataFrame(), pd.Series(dtype=object)

    long_df = pd.concat(long_records, ignore_index=True)

    # ---- tsfresh extraction on windows ----
    fc = EfficientFCParameters()
    feats_win = extract_features(
        long_df,
        column_id="id",
        column_sort="time",
        default_fc_parameters=fc,
        disable_progressbar=True,
        impute_function=None,  # we’ll clean ourselves
    ).sort_index()

    # safety: replace inf/NaN and keep numeric
    feats_win = feats_win.replace([np.inf, -np.inf], np.nan)
    feats_win = feats_win.astype(float)

    # ---- aggregate per subject: mean, std, skew, kurt over windows ----
    # map each window row -> subject
    map_df = pd.DataFrame(subj_for_window, columns=["id", "subject"]).set_index("id")
    feats_win = feats_win.join(map_df, how="left")
    agg = feats_win.groupby("subject").agg(["mean", "std", "skew", "kurt"])
    # flatten MultiIndex columns: feature__stat
    agg.columns = [f"{c[0]}__{c[1]}" for c in agg.columns]
    # final imputation by column medians
    agg = agg.replace([np.inf, -np.inf], np.nan)
    agg = agg.fillna(agg.median())
    return agg.sort_index(), pd.Series(agg.index, index=agg.index, name="subject")
