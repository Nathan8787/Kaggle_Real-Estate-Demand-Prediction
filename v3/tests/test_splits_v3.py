from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.splits_v3 import FOLD_DEFINITIONS_V3, generate_time_series_folds_v3


def _make_features_df() -> pd.DataFrame:
    months = pd.date_range("2019-01-01", "2024-07-01", freq="MS")
    records = []
    for month in months:
        records.append(
            {
                "month": month,
                "sector_id": 1,
                "amount_new_house_transactions": 100.0 + month.month,
            }
        )
    return pd.DataFrame(records)


def test_fold_alignment(tmp_path: Path):
    features_df = _make_features_df()
    reports_dir = tmp_path / "reports"
    folds = generate_time_series_folds_v3(features_df, reports_dir=reports_dir)

    assert len(folds) == len(FOLD_DEFINITIONS_V3)
    for (train_idx, valid_idx), (train_start, train_end, valid_start, valid_end) in zip(
        folds, FOLD_DEFINITIONS_V3
    ):
        assert train_idx.size > 0
        assert valid_idx.size > 0
        train_months = pd.to_datetime(features_df.loc[train_idx, "month"])
        valid_months = pd.to_datetime(features_df.loc[valid_idx, "month"])
        forecast_start = pd.Timestamp(valid_start)
        assert train_months.max() < forecast_start
        assert valid_months.min() >= forecast_start
        assert valid_months.max() <= pd.Timestamp(valid_end)

    report_path = reports_dir / "folds_definition_v3.json"
    assert report_path.exists()
    with report_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert len(payload) == len(FOLD_DEFINITIONS_V3)
    assert all("forecast_start" in entry for entry in payload)
