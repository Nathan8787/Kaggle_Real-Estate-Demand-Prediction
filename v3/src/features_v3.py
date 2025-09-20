from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .panel_builder_v3 import POI_CORE_COLUMNS, TARGET_CUTOFF
from .projection_v3 import apply_projection

logger = logging.getLogger(__name__)

__all__ = ["build_feature_matrix_v3"]

RAW_MONTHLY_COLUMNS = [
    "amount_new_house_transactions",
    "num_new_house_transactions",
    "area_new_house_transactions",
    "price_new_house_transactions",
    "area_per_unit_new_house_transactions",
    "total_price_per_unit_new_house_transactions",
    "num_new_house_available_for_sale",
    "area_new_house_available_for_sale",
    "period_new_house_sell_through",
    "amount_new_house_transactions_nearby_sectors",
    "num_new_house_transactions_nearby_sectors",
    "area_new_house_transactions_nearby_sectors",
    "price_new_house_transactions_nearby_sectors",
    "area_per_unit_new_house_transactions_nearby_sectors",
    "total_price_per_unit_new_house_transactions_nearby_sectors",
    "num_new_house_available_for_sale_nearby_sectors",
    "area_new_house_available_for_sale_nearby_sectors",
    "period_new_house_sell_through_nearby_sectors",
    "amount_pre_owned_house_transactions",
    "num_pre_owned_house_transactions",
    "area_pre_owned_house_transactions",
    "price_pre_owned_house_transactions",
    "amount_pre_owned_house_transactions_nearby_sectors",
    "num_pre_owned_house_transactions_nearby_sectors",
    "area_pre_owned_house_transactions_nearby_sectors",
    "price_pre_owned_house_transactions_nearby_sectors",
    "transaction_amount",
    "num_land_transactions",
    "construction_area",
    "planned_building_area",
    "transaction_amount_nearby_sectors",
    "num_land_transactions_nearby_sectors",
    "construction_area_nearby_sectors",
    "planned_building_area_nearby_sectors",
]

TARGET_FEATURES = [
    "target",
    "target_filled",
    "target_available_flag",
    "target_filled_was_missing",
    "is_future",
]

CITY_FEATURES = [
    "city_gdp_100m",
    "city_secondary_industry_100m",
    "city_tertiary_industry_100m",
    "city_gdp_per_capita_yuan",
    "city_total_households_10k",
    "city_year_end_resident_population_10k",
    "city_total_retail_sales_of_consumer_goods_100m",
    "city_per_capita_disposable_income_absolute_yuan",
    "city_annual_average_wage_urban_non_private_employees_yuan",
    "city_number_of_universities",
    "city_hospital_beds_10k",
    "city_number_of_operating_bus_lines",
]

CITY_FLAG_COLUMNS = [f"{col}_was_interpolated" for col in CITY_FEATURES]

METRIC_SET_FULL = RAW_MONTHLY_COLUMNS
METRIC_SET_LONG = [
    "amount_new_house_transactions",
    "num_new_house_transactions",
    "price_new_house_transactions",
    "amount_new_house_transactions_nearby_sectors",
    "num_new_house_transactions_nearby_sectors",
    "price_new_house_transactions_nearby_sectors",
    "amount_pre_owned_house_transactions",
    "num_pre_owned_house_transactions",
    "amount_pre_owned_house_transactions_nearby_sectors",
    "num_pre_owned_house_transactions_nearby_sectors",
    "transaction_amount",
    "transaction_amount_nearby_sectors",
]

GROWTH_METRICS = [
    "amount_new_house_transactions",
    "num_new_house_transactions",
    "price_new_house_transactions",
    "amount_pre_owned_house_transactions",
    "num_pre_owned_house_transactions",
    "transaction_amount",
]

SHARE_METRICS = [
    "amount_new_house_transactions",
    "num_new_house_transactions",
    "price_new_house_transactions",
    "amount_new_house_transactions_nearby_sectors",
    "num_new_house_transactions_nearby_sectors",
    "amount_pre_owned_house_transactions",
    "num_pre_owned_house_transactions",
    "transaction_amount",
    "transaction_amount_nearby_sectors",
    "num_land_transactions",
]

TIME_FEATURES = [
    "year",
    "month_num",
    "quarter",
    "month_index",
    "days_in_month",
    "is_year_start",
    "is_year_end",
]

SEARCH_VARIANCE_LIMIT = 12


def build_feature_matrix_v3(
    panel_path: Path | str,
    forecast_start: str | pd.Timestamp,
    features_path: Path | str | None,
    reports_dir: Path | str | None,
) -> tuple[pd.DataFrame, dict]:
    """Construct the v3 feature matrix following the published specification."""

    panel_path = Path(panel_path)
    features_df = pd.read_parquet(panel_path)
    features_df["month"] = pd.to_datetime(features_df["month"], errors="coerce")
    features_df = features_df.sort_values(["month", "sector_id"]).reset_index(drop=True)

    forecast_start_ts = pd.Timestamp(forecast_start).to_period("M").to_timestamp()
    features_df = apply_projection(features_df, forecast_start_ts)

    # Ensure targets after cutoff remain NaN
    features_df.loc[features_df["month"] > TARGET_CUTOFF, "target"] = np.nan

    categories: Dict[str, List[str]] = defaultdict(list)
    missing_report: List[dict] = []

    _add_time_features(features_df)
    categories["time"].extend(TIME_FEATURES)
    features_df["population_weight"] = np.where(
        features_df["resident_population"] > 0,
        features_df["resident_population"],
        np.where(
            features_df["population_scale"] > 0,
            features_df["population_scale"],
            1.0,
        ),
    )
    categories["raw"].extend([
        col for col in RAW_MONTHLY_COLUMNS if col in features_df.columns
    ])
    categories["raw"].extend(
        [col for col in TARGET_FEATURES if col in features_df.columns and col not in {"target", "target_filled_was_missing"}]
    )
    if "target_filled_was_missing" in features_df.columns:
        categories["missing_flags"].append("target_filled_was_missing")
    categories["raw"].extend([col for col in CITY_FEATURES if col in features_df.columns])
    categories["missing_flags"].extend(
        [col for col in CITY_FLAG_COLUMNS if col in features_df.columns]
    )
    categories["raw"].extend(
        [col for col in POI_CORE_COLUMNS if col in features_df.columns]
    )
    categories["raw"].append("population_weight")
    categories["raw"].append("sector_id")

    features_df = _add_missing_flags(features_df, forecast_start_ts, missing_report)
    categories["missing_flags"].extend(
        [
            f"{column}_was_missing"
            for column in RAW_MONTHLY_COLUMNS
            if f"{column}_was_missing" in features_df.columns
        ]
    )

    lag_features = _create_lag_features(features_df)
    for col in lag_features:
        categories["lag"].append(col)
    long_lag_features = _create_long_lag_features(features_df)
    for col in long_lag_features:
        categories["lag_long"].append(col)

    rolling_mean_features = _create_rolling_means(features_df)
    for col in rolling_mean_features:
        if col.endswith("_12"):
            categories["rolling_mean_long"].append(col)
        else:
            categories["rolling_mean"].append(col)

    rolling_std_features = _create_rolling_stds(features_df)
    categories["rolling_std"].extend(rolling_std_features)

    growth_features, growth_flags = _create_growth_features(features_df)
    categories["growth"].extend(growth_features)
    categories["missing_flags"].extend(growth_flags)

    share_features, weighted_features = _create_share_features(features_df)
    categories["share"].extend(share_features)
    categories["weighted_mean"].extend(weighted_features)

    log_sources = [col for col in RAW_MONTHLY_COLUMNS if col in features_df.columns]
    log_sources.extend(growth_features)
    log_sources.extend(weighted_features)
    log_sources.extend(share_features)
    log_features = _create_log1p_features(features_df, log_sources)
    categories["log1p"].extend(log_features)

    search_report_path = None
    if reports_dir is not None:
        reports_dir = Path(reports_dir)
        search_report_path = reports_dir / "search_keywords_v3.json"
    selected_search = _select_search_keywords(
        features_df, forecast_start_ts, SEARCH_VARIANCE_LIMIT, search_report_path
    )
    categories["search"].extend([col for col in selected_search if col in features_df.columns])
    search_features = _create_search_features(features_df, selected_search)
    categories["search"].extend(search_features)
    categories["missing_flags"].extend([f"{col}_was_missing" for col in selected_search])

    poi_features = _create_poi_pca(features_df, reports_dir)
    categories["poi_pca"].extend(poi_features)

    # Remove unselected search columns
    _drop_unselected_search_columns(features_df, selected_search)

    projection_meta_cols = [
        col
        for col in features_df.columns
        if col.endswith("_proj_source") or col.endswith("_proj_overridden")
    ]
    if projection_meta_cols:
        features_df.drop(columns=projection_meta_cols, inplace=True)
    projection_flag_cols = [
        col
        for col in features_df.columns
        if "_proj_source" in col or "_proj_overridden" in col
    ]
    if projection_flag_cols:
        features_df.drop(columns=projection_flag_cols, inplace=True)
    redundant_missing = [
        col for col in features_df.columns if col.endswith("_was_missing_was_missing")
    ]
    if redundant_missing:
        features_df.drop(columns=redundant_missing, inplace=True)

    if reports_dir is not None:
        reports_path = Path(reports_dir)
        reports_path.mkdir(parents=True, exist_ok=True)
        missing_path = reports_path / "missing_value_report.json"
        with missing_path.open("w", encoding="utf-8") as handle:
            json.dump(missing_report, handle, indent=2, ensure_ascii=False)

    inventory = _build_inventory(features_df, categories)
    feature_columns = inventory["feature_columns"]

    if not 537 <= len(feature_columns) <= 567:
        counts = {k: len(v) for k, v in inventory["by_category"].items()}
        categorized = set().union(*inventory["by_category"].values())
        uncategorized = sorted(set(feature_columns) - categorized)
        raise ValueError(
            f"feature count {len(feature_columns)} outside expected range; "
            f"by_category_counts={counts}; uncategorized={uncategorized}"
        )

    if features_path is not None:
        features_path = Path(features_path)
        features_path.parent.mkdir(parents=True, exist_ok=True)
        export_df = features_df.copy()
        export_df["year"] = export_df["month"].dt.year.astype(np.int16)
        export_df.to_parquet(features_path, index=False, partition_cols=["year"])

    if reports_dir is not None:
        inventory_path = Path(reports_dir) / "feature_inventory_v3.json"
        with inventory_path.open("w", encoding="utf-8") as handle:
            json.dump({k: v for k, v in inventory.items() if k != "feature_columns"}, handle, indent=2, ensure_ascii=False)

    metadata = {
        "selected_search_keywords": selected_search,
        "inventory": {k: v for k, v in inventory.items() if k != "feature_columns"},
    }

    return features_df, metadata


def _add_time_features(df: pd.DataFrame) -> None:
    df["year"] = df["month"].dt.year.astype(np.int16)
    df["month_num"] = df["month"].dt.month.astype(np.int8)
    df["quarter"] = df["month"].dt.quarter.astype(np.int8)
    start_year = df["month"].dt.year.min()
    start_month = df["month"].dt.month.min()
    df["month_index"] = (
        (df["month"].dt.year - start_year) * 12 + (df["month"].dt.month - start_month)
    ).astype(np.int32)
    df["days_in_month"] = df["month"].dt.daysinmonth.astype(np.int8)
    df["is_year_start"] = (df["month"].dt.month == 1).astype(np.int8)
    df["is_year_end"] = (df["month"].dt.month == 12).astype(np.int8)


def _add_missing_flags(
    df: pd.DataFrame,
    forecast_start: pd.Timestamp,
    report: List[dict],
) -> pd.DataFrame:
    df = df.sort_values(["sector_id", "month"]).reset_index(drop=True)

    for column in RAW_MONTHLY_COLUMNS:
        if column in df.columns and f"{column}_was_missing" not in df.columns:
            df[f"{column}_was_missing"] = df[column].isna().astype(np.int8)

    _apply_missing_policy_counts(df, RAW_MONTHLY_COLUMNS, forecast_start, report)
    _apply_missing_policy_prices(df, forecast_start, report)
    _apply_missing_policy_city(df, forecast_start, report)
    _apply_missing_policy_search(df, forecast_start, report)

    if not report:
        report.append(
            {
                "column": "__no_imputation__",
                "strategy": "none",
                "filled": 0,
                "projection_stage": "observed",
            }
        )

    return df


def _apply_missing_policy_counts(
    df: pd.DataFrame,
    columns: Iterable[str],
    forecast_start: pd.Timestamp,
    report: List[dict],
) -> None:
    for column in columns:
        if column not in df.columns:
            continue

        mask_before = df[column].isna()
        df[column] = df.groupby("sector_id")[column].ffill()
        _log_imputation(report, column, "sector_ffill", mask_before & df[column].notna(), forecast_start, df)

        mask_missing = df[column].isna()
        if mask_missing.any():
            rolling = (
                df.groupby("sector_id")[column]
                .transform(lambda s: s.shift(1).rolling(window=12, min_periods=3).median())
            )
            df.loc[mask_missing, column] = rolling.loc[mask_missing]
            _log_imputation(report, column, "rolling_median_12", mask_missing & df[column].notna(), forecast_start, df)

        mask_missing = df[column].isna()
        if mask_missing.any():
            global_median = float(df[column].median(skipna=True)) if not df[column].dropna().empty else 0.0
            df.loc[mask_missing, column] = global_median
            _log_imputation(report, column, "global_median", mask_missing & df[column].notna(), forecast_start, df)

        mask_missing = df[column].isna()
        if mask_missing.any():
            df.loc[mask_missing, column] = 0.0
            _log_imputation(report, column, "fill_zero", mask_missing, forecast_start, df)


def _apply_missing_policy_prices(
    df: pd.DataFrame,
    forecast_start: pd.Timestamp,
    report: List[dict],
) -> None:
    price_columns = [
        col
        for col in df.columns
        if (col.startswith("price_") or col.startswith("total_price_") or col.startswith("period_"))
        and "_proj_" not in col
        and not col.endswith("_was_missing")
    ]
    for column in price_columns:
        if f"{column}_was_missing" not in df.columns:
            df[f"{column}_was_missing"] = df[column].isna().astype(np.int8)

        mask_before = df[column].isna()
        df[column] = df.groupby("sector_id")[column].ffill()
        _log_imputation(report, column, "sector_ffill", mask_before & df[column].notna(), forecast_start, df)

        mask_missing = df[column].isna()
        if mask_missing.any():
            medians = df.groupby("sector_id")[column].transform("median")
            df.loc[mask_missing, column] = medians.loc[mask_missing]
            _log_imputation(report, column, "sector_median", mask_missing & df[column].notna(), forecast_start, df)

        mask_missing = df[column].isna()
        if mask_missing.any():
            global_median = float(df[column].median(skipna=True)) if not df[column].dropna().empty else 0.0
            df.loc[mask_missing, column] = global_median
            _log_imputation(report, column, "global_median", mask_missing & df[column].notna(), forecast_start, df)

        mask_missing = df[column].isna()
        if mask_missing.any():
            df.loc[mask_missing, column] = 0.0
            _log_imputation(report, column, "fill_zero", mask_missing, forecast_start, df)


def _apply_missing_policy_city(
    df: pd.DataFrame,
    forecast_start: pd.Timestamp,
    report: List[dict],
) -> None:
    for column in CITY_FEATURES:
        if column not in df.columns:
            continue
        mask_before = df[column].isna()
        df[column] = df[column].ffill()
        _log_imputation(report, column, "ffill", mask_before & df[column].notna(), forecast_start, df)

        mask_missing = df[column].isna()
        if mask_missing.any():
            month_medians = df.groupby(df["month"].dt.month)[column].transform("median")
            df.loc[mask_missing, column] = month_medians.loc[mask_missing]
            _log_imputation(report, column, "month_median", mask_missing & df[column].notna(), forecast_start, df)

        mask_missing = df[column].isna()
        if mask_missing.any():
            global_median = float(df[column].median(skipna=True)) if not df[column].dropna().empty else 0.0
            df.loc[mask_missing, column] = global_median
            _log_imputation(report, column, "global_median", mask_missing & df[column].notna(), forecast_start, df)


def _apply_missing_policy_search(
    df: pd.DataFrame,
    forecast_start: pd.Timestamp,
    report: List[dict],
) -> None:
    search_columns = [
        col
        for col in df.columns
        if col.startswith("search_kw_")
        and "_proj_" not in col
        and "_lag_" not in col
        and "_rolling_" not in col
        and "_pct_change" not in col
        and "_zscore" not in col
        and not col.endswith("_was_missing")
    ]
    for column in search_columns:
        flag_col = f"{column}_was_missing"
        if flag_col not in df.columns:
            df[flag_col] = df[column].isna().astype(np.int8)
        else:
            df.loc[df[column].isna(), flag_col] = 1
        mask_before = df[column].isna()
        if mask_before.any():
            df.loc[mask_before, column] = 0.0
            _log_imputation(report, column, "fill_zero", mask_before, forecast_start, df)


def _log_imputation(
    report: List[dict],
    column: str,
    strategy: str,
    filled_mask: pd.Series,
    forecast_start: pd.Timestamp,
    df: pd.DataFrame,
) -> None:
    if filled_mask.empty or not filled_mask.any():
        return
    observed_mask = df["month"] < forecast_start
    observed_count = int((filled_mask & observed_mask).sum())
    forecast_count = int((filled_mask & ~observed_mask).sum())
    if observed_count:
        report.append(
            {
                "column": column,
                "strategy": strategy,
                "filled": observed_count,
                "projection_stage": "observed",
            }
        )
    if forecast_count:
        report.append(
            {
                "column": column,
                "strategy": strategy,
                "filled": forecast_count,
                "projection_stage": "forecast",
            }
        )


def _create_lag_features(df: pd.DataFrame) -> List[str]:
    created = []
    for step in (1, 3, 6):
        for column in METRIC_SET_FULL:
            if column not in df.columns:
                continue
            lag_col = f"{column}_lag_{step}"
            df[lag_col] = (
                df.groupby("sector_id")[column]
                .shift(step)
                .astype(float)
            )
            df[lag_col] = df.groupby("sector_id")[lag_col].ffill()
            df[lag_col] = df[lag_col].fillna(0.0)
            created.append(lag_col)
    return created


def _create_long_lag_features(df: pd.DataFrame) -> List[str]:
    created = []
    for column in METRIC_SET_LONG:
        if column not in df.columns:
            continue
        lag_col = f"{column}_lag_12"
        df[lag_col] = df.groupby("sector_id")[column].shift(12).astype(float)
        df[lag_col] = df.groupby("sector_id")[lag_col].ffill().fillna(0.0)
        created.append(lag_col)
    return created


def _create_rolling_means(df: pd.DataFrame) -> List[str]:
    created = []
    for window in (3, 6):
        for column in METRIC_SET_FULL:
            if column not in df.columns:
                continue
            roll_col = f"{column}_rolling_mean_{window}"
            df[roll_col] = df.groupby("sector_id")[column].transform(
                lambda s: s.shift(1).rolling(window=window, min_periods=2).mean()
            )
            df[roll_col] = df.groupby("sector_id")[roll_col].ffill().fillna(0.0)
            created.append(roll_col)
    for column in METRIC_SET_LONG:
        if column not in df.columns:
            continue
        roll_col = f"{column}_rolling_mean_12"
        df[roll_col] = df.groupby("sector_id")[column].transform(
            lambda s: s.shift(1).rolling(window=12, min_periods=2).mean()
        )
        df[roll_col] = df.groupby("sector_id")[roll_col].ffill().fillna(0.0)
        created.append(roll_col)
    return created


def _create_rolling_stds(df: pd.DataFrame) -> List[str]:
    created = []
    for window in (3, 6):
        for column in METRIC_SET_FULL:
            if column not in df.columns:
                continue
            roll_col = f"{column}_rolling_std_{window}"
            df[roll_col] = df.groupby("sector_id")[column].transform(
                lambda s: s.shift(1).rolling(window=window, min_periods=2).std(ddof=0)
            )
            df[roll_col] = df.groupby("sector_id")[roll_col].ffill().fillna(0.0)
            created.append(roll_col)
    return created


def _create_growth_features(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    growth_cols: List[str] = []
    flag_cols: List[str] = []
    for metric in GROWTH_METRICS:
        if metric not in df.columns:
            continue
        for horizon in (1, 3):
            lag_col = f"{metric}_lag_{horizon}"
            if lag_col not in df.columns:
                continue
            growth_col = f"{metric}_growth_{horizon}"
            lag_values = df[lag_col]
            current_values = df[metric]
            ratio = (current_values - lag_values) / (np.abs(lag_values) + 1e-6)
            missing_flag = lag_values.isna().astype(np.int8)
            ratio = ratio.fillna(0.0)
            df[growth_col] = ratio
            df[f"{growth_col}_was_missing"] = missing_flag
            growth_cols.append(growth_col)
            flag_cols.append(f"{growth_col}_was_missing")
    return growth_cols, flag_cols


def _create_share_features(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    share_cols: List[str] = []
    weighted_cols: List[str] = []
    weight = df["population_weight"].replace(0.0, 1.0)
    for metric in SHARE_METRICS:
        if metric not in df.columns:
            continue
        weighted_col = f"{metric}_city_weighted_mean"
        share_col = f"{metric}_share"
        numerator = (df[metric] * weight)
        denom = df.groupby("month")["population_weight"].transform("sum")
        with np.errstate(invalid="ignore"):
            city_weighted = numerator.groupby(df["month"]).transform("sum") / np.where(denom == 0, 1.0, denom)
        city_weighted = city_weighted.fillna(df[metric].mean())
        df[weighted_col] = city_weighted.fillna(0.0)
        df[share_col] = df[metric] / (df[weighted_col] + 1e-6)
        df[share_col] = df[share_col].replace([np.inf, -np.inf], 0.0).fillna(0.0)
        share_cols.append(share_col)
        weighted_cols.append(weighted_col)
    return share_cols, weighted_cols


def _create_log1p_features(df: pd.DataFrame, columns: Iterable[str]) -> List[str]:
    created = []
    seen = set()
    for column in columns:
        if column not in df.columns or column in seen:
            continue
        log_col = f"{column}_log1p"
        df[log_col] = np.log1p(np.clip(df[column].astype(float), 0.0, None))
        created.append(log_col)
        seen.add(column)
    return created


def _select_search_keywords(
    df: pd.DataFrame,
    forecast_start: pd.Timestamp,
    limit: int,
    report_path: Path | None,
) -> List[str]:
    search_cols = [
        col
        for col in df.columns
        if col.startswith("search_kw_")
        and "_lag_" not in col
        and "_rolling_" not in col
        and "_pct_change" not in col
        and "_zscore" not in col
        and not col.endswith("_was_missing")
    ]
    variances = []
    observed_mask = df["month"] <= TARGET_CUTOFF
    for col in search_cols:
        var = df.loc[observed_mask, col].var(ddof=0)
        variances.append((col, 0.0 if pd.isna(var) else float(var)))
    variances.sort(key=lambda item: item[1], reverse=True)
    selected = [col for col, _ in variances[:limit]]
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump({"selected": selected, "variance": variances}, handle, indent=2, ensure_ascii=False)
    return selected


def _create_search_features(df: pd.DataFrame, selected: List[str]) -> List[str]:
    created: List[str] = []
    for col in selected:
        lag_col = f"{col}_lag_1"
        rolling_col = f"{col}_rolling_mean_3"
        pct_col = f"{col}_pct_change_1"
        z_col = f"{col}_zscore_3m"

        lag_series = df.groupby("sector_id")[col].shift(1)
        df[lag_col] = lag_series.fillna(0.0)

        df[rolling_col] = df.groupby("sector_id")[col].transform(
            lambda s: s.shift(1).rolling(window=3, min_periods=1).mean()
        )
        df[rolling_col] = df[rolling_col].fillna(0.0)

        df[pct_col] = (df[col] - df[lag_col]) / (np.abs(df[lag_col]) + 1e-6)
        df[pct_col] = df[pct_col].replace([np.inf, -np.inf], 0.0).fillna(0.0)

        mean = df.groupby("sector_id")[col].transform(
            lambda s: s.shift(1).rolling(window=3, min_periods=1).mean()
        )
        std = df.groupby("sector_id")[col].transform(
            lambda s: s.shift(1).rolling(window=3, min_periods=1).std(ddof=0)
        )
        df[z_col] = (df[col] - mean) / (std + 1e-6)
        df[z_col] = df[z_col].replace([np.inf, -np.inf], 0.0).fillna(0.0)

        created.extend([lag_col, rolling_col, pct_col, z_col])
    return created


def _drop_unselected_search_columns(df: pd.DataFrame, selected: List[str]) -> None:
    search_cols = [
        col
        for col in df.columns
        if col.startswith("search_kw_")
        and "_lag_" not in col
        and "_rolling_" not in col
        and "_pct_change" not in col
        and "_zscore" not in col
        and not col.endswith("_was_missing")
    ]
    drop_cols = [col for col in search_cols if col not in selected]
    drop_flags = [f"{col}_was_missing" for col in drop_cols]
    df.drop(columns=drop_cols + drop_flags, inplace=True, errors="ignore")


def _create_poi_pca(df: pd.DataFrame, reports_dir: Path | str | None) -> List[str]:
    poi_columns = [
        col
        for col in df.columns
        if (col.endswith("_dense") or col.startswith("number_of_"))
        and col not in POI_CORE_COLUMNS
    ]
    if not poi_columns:
        df["poi_pca_1"] = 0.0
        df["poi_pca_2"] = 0.0
        if reports_dir is not None:
            reports_path = Path(reports_dir)
            reports_path.mkdir(parents=True, exist_ok=True)
            summary = {
                "original_feature_count": 0,
                "explained_variance_ratio": [0.0, 0.0],
                "columns": [],
                "used_pca": False,
            }
            with (reports_path / "poi_pca_summary.json").open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2, ensure_ascii=False)
        return ["poi_pca_1", "poi_pca_2"]

    sector_poi = df[["sector_id", *poi_columns]].drop_duplicates("sector_id")
    matrix = sector_poi[poi_columns].fillna(0.0).astype(float)
    matrix = np.log1p(np.clip(matrix, 0.0, None))
    if matrix.shape[1] < 2:
        fallback = matrix.iloc[:, 0] if matrix.shape[1] == 1 else pd.Series(0.0, index=matrix.index)
        fallback = fallback.reindex(sector_poi.index, fill_value=0.0)
        mapped = dict(zip(sector_poi["sector_id"], fallback))
        df.drop(columns=["poi_pca_1", "poi_pca_2"], inplace=True, errors="ignore")
        df["poi_pca_1"] = df["sector_id"].map(mapped).fillna(0.0)
        df["poi_pca_2"] = 0.0
        df.drop(columns=poi_columns, inplace=True, errors="ignore")
        if reports_dir is not None:
            reports_path = Path(reports_dir)
            reports_path.mkdir(parents=True, exist_ok=True)
            summary = {
                "original_feature_count": len(poi_columns),
                "explained_variance_ratio": [1.0, 0.0],
                "columns": poi_columns,
                "used_pca": False,
            }
            with (reports_path / "poi_pca_summary.json").open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2, ensure_ascii=False)
        return ["poi_pca_1", "poi_pca_2"]

    pca = PCA(n_components=2, random_state=42)
    transformed = pca.fit_transform(matrix)
    poi_features = ["poi_pca_1", "poi_pca_2"]
    mapping = pd.DataFrame(
        {
            "sector_id": sector_poi["sector_id"],
            "poi_pca_1": transformed[:, 0],
            "poi_pca_2": transformed[:, 1],
        }
    )
    df.drop(columns=poi_features, inplace=True, errors="ignore")
    map_1 = dict(zip(mapping["sector_id"], mapping["poi_pca_1"]))
    map_2 = dict(zip(mapping["sector_id"], mapping["poi_pca_2"]))
    df["poi_pca_1"] = df["sector_id"].map(map_1).fillna(0.0)
    df["poi_pca_2"] = df["sector_id"].map(map_2).fillna(0.0)
    df.drop(columns=poi_columns, inplace=True, errors="ignore")

    if reports_dir is not None:
        reports_path = Path(reports_dir)
        reports_path.mkdir(parents=True, exist_ok=True)
        summary = {
            "original_feature_count": len(poi_columns),
            "explained_variance_ratio": list(map(float, pca.explained_variance_ratio_)),
            "columns": poi_columns,
            "used_pca": True,
        }
        with (reports_path / "poi_pca_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)

    return poi_features


def _build_inventory(df: pd.DataFrame, categories: Dict[str, List[str]]) -> dict:
    unique_categories = {key: sorted(set(value)) for key, value in categories.items()}

    feature_columns = [
        col
        for col in df.columns
        if col not in {"month", "id", "target"}
    ]
    total = len(feature_columns)

    inventory = {
        "total_features": total,
        "by_category": unique_categories,
        "feature_columns": feature_columns,
    }
    return inventory
