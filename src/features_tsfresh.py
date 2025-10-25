from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from tsfresh.feature_extraction import extract_features, EfficientFCParameters

warnings.filterwarnings("ignore", category=RuntimeWarning)


def tsfresh_from_sequences(
    seqs: dict,
    kind: str = "value",
    default_fc_parameters=None,
) -> pd.DataFrame:
    """
    seqs: dict mapping key -> 1D numpy array (all same length recommended)
    returns feature matrix (rows=keys)
    """
    if default_fc_parameters is None:
        default_fc_parameters = EfficientFCParameters()

    # Build long-form dataframe for tsfresh
    records = []
    for key, arr in seqs.items():
        arr = np.asarray(arr)
        idx = np.arange(len(arr))
        df_key = pd.DataFrame({"id": key, "time": idx, kind: arr})
        records.append(df_key)
    long_df = pd.concat(records, ignore_index=True)

    feats = extract_features(
        long_df,
        column_id="id",
        column_sort="time",
        default_fc_parameters=default_fc_parameters,
        disable_progressbar=True,
        impute_function=None,  # we'll handle NaN later
    ).sort_index()

    # simple imputation: replace remaining NaNs/inf with column medians
    feats = feats.replace([np.inf, -np.inf], np.nan)
    feats = feats.fillna(feats.median())
    return feats
