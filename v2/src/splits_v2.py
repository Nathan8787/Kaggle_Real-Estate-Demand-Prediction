from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

__all__ = ["FOLD_DEFINITIONS_V2", "generate_time_series_folds_v2"]

FOLD_DEFINITIONS_V2 = [
    ("2019-01-01", "2022-12-31", "2023-01-01", "2023-03-31"),
    ("2019-01-01", "2023-03-31", "2023-04-01", "2023-06-30"),
    ("2019-01-01", "2023-06-30", "2023-07-01", "2023-09-30"),
    ("2019-01-01", "2023-09-30", "2023-10-01", "2023-12-31"),
    ("2019-01-01", "2023-12-31", "2024-01-01", "2024-03-31"),
    ("2019-01-01", "2024-03-31", "2024-04-01", "2024-05-31"),
    ("2019-01-01", "2024-05-31", "2024-06-01", "2024-07-31"),
]


def generate_time_series_folds_v2(features_df: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
    if "month" not in features_df.columns or "sector_id" not in features_df.columns:
        raise KeyError("features_df must contain 'month' and 'sector_id' columns")

    ordered = features_df.sort_values(["month", "sector_id"]).reset_index(drop=False)
    month_values = pd.to_datetime(ordered["month"])

    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for train_start, train_end, val_start, val_end in FOLD_DEFINITIONS_V2:
        train_start_ts = pd.Timestamp(train_start)
        train_end_ts = pd.Timestamp(train_end)
        val_start_ts = pd.Timestamp(val_start)
        val_end_ts = pd.Timestamp(val_end)

        train_mask = (month_values >= train_start_ts) & (month_values <= train_end_ts)
        val_mask = (month_values >= val_start_ts) & (month_values <= val_end_ts)

        train_idx = ordered.loc[train_mask, "index"].to_numpy(dtype=int)
        valid_idx = ordered.loc[val_mask, "index"].to_numpy(dtype=int)
        folds.append((train_idx, valid_idx))
    return folds
