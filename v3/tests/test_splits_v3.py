from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.splits_v3 import (
    FOLD_DEFINITIONS_V3,
    PURGE_PERIOD,
    generate_time_series_folds_v3,
)


def _load_real_features(sector_ids=(1,)) -> pd.DataFrame:
    root = Path(__file__).resolve().parents[2]
    new_house_path = root / "train" / "new_house_transactions.csv"
    sector_map_path = root / "v3" / "config" / "sector_city_map_v3.csv"

    transactions = pd.read_csv(new_house_path, encoding="utf-8-sig")
    transactions["month"] = pd.to_datetime(transactions["month"], format="%Y-%b")
    transactions["sector_id"] = (
        transactions["sector"].str.extract(r"(\d+)").astype(int)
    )
    filtered = transactions[transactions["sector_id"].isin(sector_ids)].copy()

    sector_map = pd.read_csv(sector_map_path, encoding="utf-8-sig")
    merged = filtered.merge(sector_map, on="sector", how="left", validate="many_to_one")
    merged["city_id"] = merged["city_id"].astype("int16")
    return merged.drop(columns=["sector"])


def test_fold_alignment(tmp_path: Path):
    features_df = _load_real_features()
    reports_dir = tmp_path / "reports"
    folds = generate_time_series_folds_v3(features_df, reports_dir=reports_dir)

    assert len(folds) == len(FOLD_DEFINITIONS_V3)
    for (train_idx, valid_idx), (_, _, valid_start, valid_end) in zip(
        folds, FOLD_DEFINITIONS_V3
    ):
        assert train_idx.size > 0
        assert valid_idx.size > 0
        train_months = pd.to_datetime(features_df.loc[train_idx, "month"])
        valid_months = pd.to_datetime(features_df.loc[valid_idx, "month"])
        forecast_start = pd.Timestamp(valid_start)
        purge_start = (forecast_start - PURGE_PERIOD).to_period("M").to_timestamp()

        if not train_months.empty:
            assert (train_months < forecast_start).all()
            assert not train_months.between(
                purge_start, forecast_start - pd.Timedelta(days=1)
            ).any()
        if not valid_months.empty:
            assert valid_months.min() >= forecast_start
            assert valid_months.max() <= pd.Timestamp(valid_end)

    report_path = reports_dir / "folds_definition_v3.json"
    assert report_path.exists()
    with report_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert len(payload) == len(FOLD_DEFINITIONS_V3)
    assert all(entry.get("purge_period_months") == 1 for entry in payload)
