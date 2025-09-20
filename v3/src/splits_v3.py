from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from .projection_v3 import apply_projection

logger = logging.getLogger(__name__)

__all__ = [
    "FOLD_DEFINITIONS_V3",
    "HOLDOUT_WINDOW",
    "generate_time_series_folds_v3",
]

FOLD_DEFINITIONS_V3 = [
    ("2019-01-01", "2023-11-30", "2023-12-01", "2024-01-31"),
    ("2019-01-01", "2024-01-31", "2024-02-01", "2024-03-31"),
    ("2019-01-01", "2024-03-31", "2024-04-01", "2024-05-31"),
    ("2019-01-01", "2024-04-30", "2024-05-01", "2024-06-30"),
]

HOLDOUT_WINDOW = ("2019-01-01", "2024-06-30", "2024-07-01", "2024-07-31")


def generate_time_series_folds_v3(
    features_df: pd.DataFrame,
    reports_dir: Path | str | None = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate cross-validation folds aligned with the v3 projection policy."""

    if "month" not in features_df.columns or "sector_id" not in features_df.columns:
        raise KeyError("features_df must contain 'month' and 'sector_id' columns")

    ordered = features_df.sort_values(["month", "sector_id"]).reset_index(drop=False)
    ordered["month"] = pd.to_datetime(ordered["month"], errors="coerce")

    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    records = []

    for train_start, train_end, valid_start, valid_end in FOLD_DEFINITIONS_V3:
        train_start_ts = pd.Timestamp(train_start)
        train_end_ts = pd.Timestamp(train_end)
        valid_start_ts = pd.Timestamp(valid_start)
        valid_end_ts = pd.Timestamp(valid_end)
        forecast_start = valid_start_ts

        projected = apply_projection(ordered.copy(), forecast_start)

        train_mask = (
            (projected["month"] >= train_start_ts)
            & (projected["month"] <= train_end_ts)
            & (projected["month"] < forecast_start)
        )
        valid_mask = (
            (projected["month"] >= valid_start_ts)
            & (projected["month"] <= valid_end_ts)
            & (projected["month"] >= forecast_start)
        )

        train_idx = projected.loc[train_mask, "index"].to_numpy(dtype=int)
        valid_idx = projected.loc[valid_mask, "index"].to_numpy(dtype=int)

        folds.append((train_idx, valid_idx))
        records.append(
            {
                "train_start": train_start,
                "train_end": train_end,
                "valid_start": valid_start,
                "valid_end": valid_end,
                "forecast_start": forecast_start.strftime("%Y-%m-%d"),
                "train_size": int(train_idx.size),
                "valid_size": int(valid_idx.size),
            }
        )

    if reports_dir is not None:
        reports_path = Path(reports_dir)
        reports_path.mkdir(parents=True, exist_ok=True)
        with (reports_path / "folds_definition_v3.json").open("w", encoding="utf-8") as handle:
            json.dump(records, handle, indent=2, ensure_ascii=False)

    return folds
