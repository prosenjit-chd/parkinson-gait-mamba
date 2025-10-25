from __future__ import annotations
import numpy as np
import pandas as pd


def standardize_length(x: np.ndarray, max_len: int = 12000) -> np.ndarray:
    """
    Truncate long sequences to 'max_len' (default 120s at 100 Hz),
    or pad with last value if shorter. Keeps things comparable.
    """
    if x.shape[0] >= max_len:
        return x[:max_len]
    if x.size == 0:
        return np.zeros(max_len)
    pad = np.full(max_len - x.shape[0], x[-1])
    return np.concatenate([x, pad])


def normalize_force(x: np.ndarray) -> np.ndarray:
    """Min-max scale each sequence to [0, 1] to reduce inter-subject scale effects."""
    if len(x) == 0:
        return x
    mn, mx = np.nanmin(x), np.nanmax(x)
    if mx > mn:
        return (x - mn) / (mx - mn)
    return np.zeros_like(x)
