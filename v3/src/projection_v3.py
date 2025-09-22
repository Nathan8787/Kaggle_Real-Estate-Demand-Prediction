from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .panel_builder_v3 import CITY_INDEX_ALLOWED_COLUMNS

logger = logging.getLogger(__name__)

__all__ = [
    "PROJECTION_COLUMNS",
    "apply_projection",
    "collect_projection_drift",
]

NEW_HOUSE_COLUMNS = [
    "amount_new_house_transactions",
    "num_new_house_transactions",
    "area_new_house_transactions",
    "price_new_house_transactions",
    "area_per_unit_new_house_transactions",
    "total_price_per_unit_new_house_transactions",
    "num_new_house_available_for_sale",
    "area_new_house_available_for_sale",
    "period_new_house_sell_through",
]

PREOWNED_COLUMNS = [
    "amount_pre_owned_house_transactions",
    "num_pre_owned_house_transactions",
    "area_pre_owned_house_transactions",
    "price_pre_owned_house_transactions",
]

LAND_COLUMNS = [
    "transaction_amount",
    "num_land_transactions",
    "construction_area",
    "planned_building_area",
]

MONTHLY_PROJECTION_COLUMNS = (
    NEW_HOUSE_COLUMNS
    + [f"{col}_nearby_sectors" for col in NEW_HOUSE_COLUMNS]
    + PREOWNED_COLUMNS
    + [f"{col}_nearby_sectors" for col in PREOWNED_COLUMNS]
    + LAND_COLUMNS
    + [f"{col}_nearby_sectors" for col in LAND_COLUMNS]
)

PROJECTION_COLUMNS = list(MONTHLY_PROJECTION_COLUMNS) + list(CITY_INDEX_ALLOWED_COLUMNS)


def _list_projection_columns(panel_df: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for name in PROJECTION_COLUMNS:
        if name in panel_df.columns:
            columns.append(name)

    search_candidates = [
        col
        for col in panel_df.columns
        if col.startswith("search_kw_")
        and "_lag_" not in col
        and "_rolling_" not in col
        and "_pct_change" not in col
        and "_zscore" not in col
        and not col.endswith("_was_missing")
        and not col.endswith("_proj_source")
        and not col.endswith("_proj_overridden")
    ]
    columns.extend(sorted(search_candidates))
    return columns


def apply_projection(panel_df: pd.DataFrame, forecast_start: pd.Timestamp) -> pd.DataFrame:
    """Apply forward-looking projections to the specified feature columns."""

    if "month" not in panel_df.columns:
        raise KeyError("panel_df must contain a 'month' column")
    if "sector_id" not in panel_df.columns:
        raise KeyError("panel_df must contain a 'sector_id' column")
    if "city_id" not in panel_df.columns:
        raise KeyError("panel_df must contain a 'city_id' column")

    forecast_start = pd.Timestamp(forecast_start).to_period("M").to_timestamp()
    projected = panel_df.copy()
    projected["month"] = pd.to_datetime(projected["month"], errors="coerce")
    projected = projected.sort_values(["month", "sector_id"]).reset_index(drop=True)

    projection_columns = _list_projection_columns(projected)
    for column in projection_columns:
        projected[f"{column}_proj_source"] = np.zeros(len(projected), dtype=np.int8)
        projected[f"{column}_proj_overridden"] = np.zeros(len(projected), dtype=np.int8)

    if not projection_columns:
        logger.warning("No projection columns detected for forecast_start=%s", forecast_start)
        return projected

    for column in projection_columns:
        history_mask = (projected["month"] < forecast_start) & projected[column].notna()
        history_values_all = projected.loc[history_mask, column].astype(float)
        global_median = (
            float(history_values_all.median()) if not history_values_all.empty else np.nan
        )

        city_month_median: dict[tuple[int, int], float] = {}
        city_month_count: dict[tuple[int, int], int] = {}
        if history_mask.any():
            tmp = projected.loc[history_mask, ["city_id", "month", column]].copy()
            tmp["month_num"] = tmp["month"].dt.month.astype(int)
            grouped = tmp.groupby(["city_id", "month_num"])
            city_month_median = grouped[column].median().to_dict()
            city_month_count = grouped[column].count().to_dict()

        sector_groups = projected.groupby("sector_id", sort=False).groups
        for sector_id, indices in sector_groups.items():
            sector_idx = sorted(indices, key=lambda idx: projected.at[idx, "month"])
            sector_history = history_values_all[projected.loc[history_mask, "sector_id"] == sector_id]
            history_list = sector_history.tolist()

            for idx in sector_idx:
                month_value = projected.at[idx, "month"]
                if pd.isna(month_value) or month_value < forecast_start:
                    continue

                original_value = projected.at[idx, column]
                city_id = int(projected.at[idx, "city_id"])
                projected_value, source_code = _project_value(
                    history_list,
                    month_value,
                    city_id,
                    city_month_median,
                    city_month_count,
                    global_median,
                )

                projected_value = float(np.clip(projected_value, 0.0, None))
                if pd.notna(original_value):
                    projected.at[idx, f"{column}_proj_overridden"] = 1
                projected.at[idx, column] = projected_value
                projected.at[idx, f"{column}_proj_source"] = source_code

    return projected


def _project_value(
    history_values: Iterable[float],
    month_value: pd.Timestamp,
    city_id: int,
    city_month_median: dict[tuple[int, int], float],
    city_month_count: dict[tuple[int, int], int],
    global_median: float | np.floating | float,
) -> tuple[float, int]:
    history = [float(v) for v in history_values if pd.notna(v)]
    n_obs = len(history)

    if n_obs >= 12:
        recent = np.array(history[-12:], dtype=float)
        weights = 0.7 ** np.arange(len(recent))[::-1]
        weights = weights / weights.sum()
        value = float(np.dot(recent, weights))
        return value, 1

    if 6 <= n_obs < 12:
        value = float(np.mean(history)) if history else 0.0
        return value, 2

    if 3 <= n_obs < 6:
        value = float(np.median(history)) if history else 0.0
        return value, 2

    month_key = int(month_value.month)
    lookup_key = (city_id, month_key)
    if city_month_count.get(lookup_key, 0) >= 3:
        return float(city_month_median.get(lookup_key, 0.0)), 3

    if global_median is not None and not np.isnan(global_median):
        return float(global_median), 4

    return 0.0, 4


def collect_projection_drift(
    original_df: pd.DataFrame,
    projected_df: pd.DataFrame,
    forecast_start: pd.Timestamp,
    reports_dir: Path | None = None,
) -> dict:
    """Compute projection drift diagnostics and optionally persist them."""

    forecast_start = pd.Timestamp(forecast_start).to_period("M").to_timestamp()
    columns = _list_projection_columns(projected_df)
    diagnostics: dict[str, dict[str, float]] = {}
    actual_available = False

    for column in columns:
        mask = (
            (projected_df["month"] >= forecast_start)
            & original_df[column].notna()
            & projected_df[column].notna()
        )
        if not mask.any():
            continue

        actual_available = True
        actual = original_df.loc[mask, column].astype(float).to_numpy()
        predicted = projected_df.loc[mask, column].astype(float).to_numpy()
        error = predicted - actual
        mae = float(np.mean(np.abs(error)))
        mape = float(np.mean(np.abs(error) / (np.maximum(np.abs(actual), 1.0))))
        rmse = float(np.sqrt(np.mean(error**2)))
        p10 = float(np.percentile(error, 10))
        p90 = float(np.percentile(error, 90))
        latest_month = projected_df.loc[mask, "month"].max()

        source_col = f"{column}_proj_source"
        source_counts = projected_df.loc[mask, source_col].value_counts(normalize=True)
        diagnostics[column] = {
            "count": int(mask.sum()),
            "mae": mae,
            "mape": mape,
            "rmse": rmse,
            "p10": p10,
            "p90": p90,
            "proj_source_3_ratio": float(source_counts.get(3, 0.0)),
            "proj_source_4_ratio": float(source_counts.get(4, 0.0)),
            "latest_month": latest_month.strftime("%Y-%m-%d"),
        }

    result = {
        "forecast_start": forecast_start.strftime("%Y-%m-%d"),
        "actual_available": actual_available,
        "metrics": diagnostics,
    }

    if reports_dir is not None:
        reports_dir = Path(reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / f"projection_drift_{forecast_start.strftime('%Y%m')}.json"
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)

    return result
