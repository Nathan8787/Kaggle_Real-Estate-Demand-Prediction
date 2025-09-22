from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression

from .panel_builder_v3 import POI_CORE_COLUMNS, TARGET_CUTOFF
from .projection_v3 import (
    PROJECTION_COLUMNS,
    _project_value,
    apply_projection,
    collect_projection_drift,
)

logger = logging.getLogger(__name__)

__all__ = ["build_feature_matrix_v3"]

# fmt: off
POI_SUPPLEMENTAL_COLUMNS = [
    "bus_station_cnt",
    "catering",
    "commercial_area",
    "education",
    "education_training_school_education_kindergarten",
    "education_training_school_education_middle_school",
    "education_training_school_education_primary_school",
    "education_training_school_education_research_institution",
    "hotel",
    "leisure_and_entertainment",
    "leisure_entertainment_cultural_venue_cultural_palace",
    "leisure_entertainment_entertainment_venue_game_arcade",
    "leisure_entertainment_entertainment_venue_party_house",
    "medical_health",
    "medical_health_blood_donation_station",
    "medical_health_clinic",
    "medical_health_disease_prevention_institution",
    "medical_health_first_aid_center",
    "medical_health_general_hospital",
    "medical_health_pharmaceutical_healthcare",
    "medical_health_physical_examination_institution",
    "medical_health_rehabilitation_institution",
    "medical_health_specialty_hospital",
    "medical_health_tcm_hospital",
    "medical_health_veterinary_station",
    "office_building",
    "office_building_industrial_building_industrial_building",
    "rentable_shops",
    "residential_area",
    "retail",
    "subway_station_cnt",
    "transportation_facilities_service_airport_related",
    "transportation_facilities_service_bus_station",
    "transportation_facilities_service_light_rail_station",
    "transportation_facilities_service_long_distance_bus_station",
    "transportation_facilities_service_port_terminal",
    "transportation_facilities_service_subway_station",
    "transportation_facilities_service_train_station",
    "transportation_station",
]
# fmt: on

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

FEATURE_COUNT_MIN = 450
FEATURE_COUNT_MAX = 650


def build_feature_matrix_v3(
    panel_path: Path | str,
    forecast_start: str | pd.Timestamp,
    features_path: Path | str | None,
    reports_dir: Path | str | None,
) -> tuple[pd.DataFrame, dict]:
    """Construct the v3 feature matrix following the published specification."""

    panel_path = Path(panel_path)
    reports_path = Path(reports_dir) if reports_dir is not None else None

    features_df = pd.read_parquet(panel_path)
    features_df["month"] = pd.to_datetime(features_df["month"], errors="coerce")
    features_df = features_df.sort_values(["month", "sector_id"]).reset_index(drop=True)

    original_df = features_df.copy(deep=True)
    forecast_start_ts = pd.Timestamp(forecast_start).to_period("M").to_timestamp()
    features_df = apply_projection(features_df, forecast_start_ts)

    drift_report = collect_projection_drift(original_df, features_df, forecast_start_ts, reports_path)

    features_df.loc[features_df["month"] > TARGET_CUTOFF, "target"] = np.nan
    history_mask = features_df["month"] < forecast_start_ts

    categories: Dict[str, List[str]] = defaultdict(list)
    missing_report: List[dict] = []
    suppressible: Dict[str, set[str]] = defaultdict(set)

    _add_time_features(features_df)
    categories["time"].extend(TIME_FEATURES)

    base_raw_cols = [col for col in RAW_MONTHLY_COLUMNS if col in features_df.columns]
    categories["raw"].extend(base_raw_cols)
    categories["raw"].extend(
        [col for col in TARGET_FEATURES if col in features_df.columns]
    )
    categories["raw"].extend([col for col in CITY_FEATURES if col in features_df.columns])
    categories["missing_flags"].extend(
        [col for col in CITY_FLAG_COLUMNS if col in features_df.columns]
    )
    categories["raw"].extend([col for col in POI_CORE_COLUMNS if col in features_df.columns])
    for col in ("sector_id", "city_id"):
        if col in features_df.columns:
            categories["raw"].append(col)

    _apply_missing_policies(features_df, forecast_start_ts, missing_report)
    categories["missing_flags"].extend(
        [
            col
            for col in features_df.columns
            if col.endswith("_was_missing") and col not in categories["missing_flags"]
        ]
    )

    target_lags, target_lag_flags = _create_target_lags(features_df)
    categories["target_lag"].extend(target_lags)
    categories["target_lag_flags"].extend(target_lag_flags)

    lag_features = _create_metric_lags(features_df, history_mask, suppressible)
    categories["lag"].extend(lag_features["short"])
    categories["lag_long"].extend(lag_features["long"])

    rolling_means = _create_rolling_means(features_df, history_mask, suppressible)
    categories["rolling_mean"].extend(rolling_means["short"])
    categories["rolling_mean_long"].extend(rolling_means["long"])

    rolling_stds = _create_rolling_stds(features_df, history_mask, suppressible)
    categories["rolling_std"].extend(rolling_stds)

    growth_features, growth_flags = _create_growth_features(features_df, suppressible)
    categories["growth"].extend(growth_features)
    categories["missing_flags"].extend(growth_flags)

    share_features, weighted_features = _create_share_features(features_df, suppressible)
    categories["share"].extend(share_features)
    categories["weighted_mean"].extend(weighted_features)

    selected_search, search_report = _select_search_keywords(
        features_df, forecast_start_ts, reports_path
    )
    categories["search"].extend(selected_search)
    search_features = _create_search_features(features_df, selected_search, suppressible)
    categories["search"].extend(search_features)
    categories["missing_flags"].extend(
        [f"{col}_was_missing" for col in selected_search if f"{col}_was_missing" in features_df.columns]
    )

    poi_features, poi_summary = _create_poi_pca(features_df, history_mask)
    categories["poi_pca"].extend(poi_features)

    _drop_unselected_search_columns(features_df, selected_search)

    projection_meta_cols = [
        col
        for col in features_df.columns
        if col.endswith("_proj_source") or col.endswith("_proj_overridden")
    ]
    if projection_meta_cols:
        features_df.drop(columns=projection_meta_cols, inplace=True)

    log_sources = _collect_log_sources(categories)
    log_features = _create_log1p_features(features_df, log_sources, suppressible)
    categories["log1p"].extend(log_features)

    projection_guard = _enforce_projection_guard(
        features_df,
        suppressible,
        drift_report,
        original_df,
        forecast_start_ts,
    )

    trimmed_log1p = _enforce_feature_budget(features_df, categories, log_features)
    if trimmed_log1p:
        categories["log1p"] = [
            col for col in categories["log1p"] if col in features_df.columns
        ]

    if reports_path is not None:
        reports_path.mkdir(parents=True, exist_ok=True)
        missing_path = reports_path / "missing_value_report.json"
        with missing_path.open("w", encoding="utf-8") as handle:
            json.dump(missing_report or [{"column": "__no_imputation__", "strategy": "none", "filled": 0, "projection_stage": "observed"}], handle, indent=2, ensure_ascii=False)

        search_path = reports_path / "search_keywords_v3.json"
        with search_path.open("w", encoding="utf-8") as handle:
            json.dump(search_report, handle, indent=2, ensure_ascii=False)

        poi_path = reports_path / "poi_pca_summary.json"
        with poi_path.open("w", encoding="utf-8") as handle:
            json.dump(poi_summary, handle, indent=2, ensure_ascii=False)

    inventory = _build_inventory(
        features_df, categories, projection_guard, trimmed_log1p
    )

    if not FEATURE_COUNT_MIN <= inventory["total_features"] <= FEATURE_COUNT_MAX:
        raise ValueError(
            f"feature count {inventory['total_features']} outside expected range {FEATURE_COUNT_MIN}-{FEATURE_COUNT_MAX}"
        )

    if features_path is not None:
        features_path = Path(features_path)
        features_path.parent.mkdir(parents=True, exist_ok=True)
        export_df = features_df.copy()
        export_df["year"] = export_df["month"].dt.year.astype(np.int16)
        export_df.to_parquet(features_path, index=False, partition_cols=["year"])

    if reports_path is not None:
        inventory_path = reports_path / "feature_inventory_v3.json"
        with inventory_path.open("w", encoding="utf-8") as handle:
            json.dump(inventory, handle, indent=2, ensure_ascii=False)

    metadata = {
        "selected_search_keywords": selected_search,
        "inventory": inventory,
    }

    return features_df, metadata


def _add_time_features(df: pd.DataFrame) -> None:
    df["year"] = df["month"].dt.year.astype(np.int16)
    df["month_num"] = df["month"].dt.month.astype(np.int8)
    df["quarter"] = df["month"].dt.quarter.astype(np.int8)
    start = df["month"].min()
    df["month_index"] = (
        (df["month"].dt.year - start.year) * 12 + (df["month"].dt.month - start.month)
    ).astype(np.int32)
    df["days_in_month"] = df["month"].dt.daysinmonth.astype(np.int8)
    df["is_year_start"] = (df["month"].dt.month == 1).astype(np.int8)
    df["is_year_end"] = (df["month"].dt.month == 12).astype(np.int8)


def _apply_missing_policies(
    df: pd.DataFrame,
    forecast_start: pd.Timestamp,
    report: List[dict],
) -> None:
    history_mask = df["month"] < forecast_start

    for column in RAW_MONTHLY_COLUMNS:
        if column not in df.columns:
            continue
        flag_col = f"{column}_was_missing"
        if flag_col not in df.columns:
            df[flag_col] = df[column].isna().astype(np.int8)
        _impute_monthly_column(df, column, history_mask, report, forecast_start)

    _impute_price_columns(df, history_mask, report, forecast_start)
    _impute_city_columns(df, history_mask, report, forecast_start)
    _impute_search_columns(df, report, forecast_start)


def _impute_monthly_column(
    df: pd.DataFrame,
    column: str,
    history_mask: pd.Series,
    report: List[dict],
    forecast_start: pd.Timestamp,
) -> None:
    mask_missing = history_mask & df[column].isna()
    if mask_missing.any():
        filled = df.groupby("sector_id")[column].ffill()
        df.loc[mask_missing, column] = filled.loc[mask_missing]
        _log_imputation(report, column, "sector_ffill", mask_missing, df, forecast_start)

    mask_missing = history_mask & df[column].isna()
    if mask_missing.any():
        rolling = df.groupby("sector_id")[column].transform(
            lambda s: s.shift(1).rolling(window=12, min_periods=3).median()
        )
        df.loc[mask_missing, column] = rolling.loc[mask_missing]
        _log_imputation(report, column, "rolling_median_12", mask_missing, df, forecast_start)

    mask_missing = history_mask & df[column].isna()
    if mask_missing.any():
        global_median = df.loc[history_mask, column].median(skipna=True)
        if pd.isna(global_median):
            global_median = 0.0
        df.loc[mask_missing, column] = float(global_median)
        _log_imputation(report, column, "global_median", mask_missing, df, forecast_start)

    mask_missing = history_mask & df[column].isna()
    if mask_missing.any():
        df.loc[mask_missing, column] = 0.0
        _log_imputation(report, column, "fill_zero", mask_missing, df, forecast_start)


def _impute_price_columns(
    df: pd.DataFrame,
    history_mask: pd.Series,
    report: List[dict],
    forecast_start: pd.Timestamp,
) -> None:
    price_columns = [
        col
        for col in df.columns
        if (col.startswith("price_") or col.startswith("total_price_") or col.startswith("period_"))
        and not col.endswith("_was_missing")
    ]
    for column in price_columns:
        flag_col = f"{column}_was_missing"
        if flag_col not in df.columns:
            df[flag_col] = df[column].isna().astype(np.int8)

        mask_missing = history_mask & df[column].isna()
        if mask_missing.any():
            filled = df.groupby("sector_id")[column].ffill()
            df.loc[mask_missing, column] = filled.loc[mask_missing]
            _log_imputation(report, column, "sector_ffill", mask_missing, df, forecast_start)

        mask_missing = history_mask & df[column].isna()
        if mask_missing.any():
            median_map = (
                df.loc[history_mask].groupby("sector_id")[column].median().to_dict()
            )
            df.loc[mask_missing, column] = df.loc[mask_missing, "sector_id"].map(median_map)
            _log_imputation(report, column, "sector_median", mask_missing & df[column].notna(), df, forecast_start)

        mask_missing = history_mask & df[column].isna()
        if mask_missing.any():
            global_median = df.loc[history_mask, column].median(skipna=True)
            if pd.isna(global_median):
                global_median = 0.0
            df.loc[mask_missing, column] = float(global_median)
            _log_imputation(report, column, "global_median", mask_missing, df, forecast_start)

        mask_missing = history_mask & df[column].isna()
        if mask_missing.any():
            df.loc[mask_missing, column] = 0.0
            _log_imputation(report, column, "fill_zero", mask_missing, df, forecast_start)


def _impute_city_columns(
    df: pd.DataFrame,
    history_mask: pd.Series,
    report: List[dict],
    forecast_start: pd.Timestamp,
) -> None:
    for column in CITY_FEATURES:
        if column not in df.columns:
            continue
        mask_missing = history_mask & df[column].isna()
        if mask_missing.any():
            filled = df[column].ffill()
            df.loc[mask_missing, column] = filled.loc[mask_missing]
            _log_imputation(report, column, "ffill", mask_missing, df, forecast_start)

        mask_missing = history_mask & df[column].isna()
        if mask_missing.any():
            month_median = df.groupby(df["month"].dt.month)[column].transform("median")
            df.loc[mask_missing, column] = month_median.loc[mask_missing]
            _log_imputation(report, column, "month_median", mask_missing & df[column].notna(), df, forecast_start)

        mask_missing = history_mask & df[column].isna()
        if mask_missing.any():
            global_median = df.loc[history_mask, column].median(skipna=True)
            if pd.isna(global_median):
                global_median = 0.0
            df.loc[mask_missing, column] = float(global_median)
            _log_imputation(report, column, "global_median", mask_missing, df, forecast_start)


def _impute_search_columns(
    df: pd.DataFrame,
    report: List[dict],
    forecast_start: pd.Timestamp,
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
        mask_missing = df[column].isna()
        if mask_missing.any():
            df.loc[mask_missing, column] = 0.0
            _log_imputation(report, column, "fill_zero", mask_missing, df, forecast_start)


def _log_imputation(
    report: List[dict],
    column: str,
    strategy: str,
    filled_mask: pd.Series,
    df: pd.DataFrame,
    forecast_start: pd.Timestamp,
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


def _create_target_lags(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    lag_cols: List[str] = []
    flag_cols: List[str] = []
    for step in (1, 3, 6, 12):
        lag_col = f"target_lag_{step}"
        lag_series = df.groupby("sector_id")["target_filled"].shift(step)
        flag = lag_series.isna().astype(np.int8)
        df[lag_col] = lag_series.fillna(0.0)
        flag_col = f"{lag_col}_was_imputed"
        df[flag_col] = flag
        lag_cols.append(lag_col)
        flag_cols.append(flag_col)
    return lag_cols, flag_cols


def _create_metric_lags(
    df: pd.DataFrame,
    history_mask: pd.Series,
    suppressible: Dict[str, set[str]],
) -> Dict[str, List[str]]:
    short: List[str] = []
    long: List[str] = []
    for step in (1, 3, 6):
        for column in METRIC_SET_FULL:
            if column not in df.columns:
                continue
            lag_col = f"{column}_lag_{step}"
            lag_series = df.groupby("sector_id")[column].shift(step)
            df[lag_col] = lag_series
            df.loc[history_mask, lag_col] = df.loc[history_mask, lag_col].ffill().fillna(0.0)
            short.append(lag_col)
    for column in METRIC_SET_LONG:
        if column not in df.columns:
            continue
        lag_col = f"{column}_lag_12"
        lag_series = df.groupby("sector_id")[column].shift(12)
        df[lag_col] = lag_series
        df.loc[history_mask, lag_col] = df.loc[history_mask, lag_col].ffill().fillna(0.0)
        long.append(lag_col)
    return {"short": short, "long": long}


def _create_rolling_means(
    df: pd.DataFrame,
    history_mask: pd.Series,
    suppressible: Dict[str, set[str]],
) -> Dict[str, List[str]]:
    short: List[str] = []
    long: List[str] = []
    for window in (3, 6):
        for column in METRIC_SET_FULL:
            if column not in df.columns:
                continue
            roll_col = f"{column}_rolling_mean_{window}"
            series = df.groupby("sector_id")[column].transform(
                lambda s: s.shift(1).rolling(window=window, min_periods=2).mean()
            )
            df[roll_col] = series
            df.loc[history_mask, roll_col] = df.loc[history_mask, roll_col].ffill().fillna(0.0)
            short.append(roll_col)
            suppressible[column].add(roll_col)
    for column in METRIC_SET_LONG:
        if column not in df.columns:
            continue
        roll_col = f"{column}_rolling_mean_12"
        series = df.groupby("sector_id")[column].transform(
            lambda s: s.shift(1).rolling(window=12, min_periods=2).mean()
        )
        df[roll_col] = series
        df.loc[history_mask, roll_col] = df.loc[history_mask, roll_col].ffill().fillna(0.0)
        long.append(roll_col)
        suppressible[column].add(roll_col)
    return {"short": short, "long": long}


def _create_rolling_stds(
    df: pd.DataFrame,
    history_mask: pd.Series,
    suppressible: Dict[str, set[str]],
) -> List[str]:
    created: List[str] = []
    for window in (3, 6):
        for column in METRIC_SET_FULL:
            if column not in df.columns:
                continue
            roll_col = f"{column}_rolling_std_{window}"
            series = df.groupby("sector_id")[column].transform(
                lambda s: s.shift(1).rolling(window=window, min_periods=2).std(ddof=0)
            )
            df[roll_col] = series
            df.loc[history_mask, roll_col] = df.loc[history_mask, roll_col].ffill().fillna(0.0)
            created.append(roll_col)
            suppressible[column].add(roll_col)
    return created


def _create_growth_features(
    df: pd.DataFrame,
    suppressible: Dict[str, set[str]],
) -> Tuple[List[str], List[str]]:
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
            ratio = (df[metric] - lag_values) / (np.abs(lag_values) + 1e-6)
            ratio = ratio.replace([np.inf, -np.inf], 0.0).fillna(0.0)
            df[growth_col] = ratio
            df[f"{growth_col}_was_missing"] = lag_values.isna().astype(np.int8)
            growth_cols.append(growth_col)
            flag_cols.append(f"{growth_col}_was_missing")
            suppressible[metric].add(growth_col)
    return growth_cols, flag_cols


def _create_share_features(
    df: pd.DataFrame,
    suppressible: Dict[str, set[str]],
) -> Tuple[List[str], List[str]]:
    share_cols: List[str] = []
    weighted_cols: List[str] = []
    if "resident_population" not in df.columns or "population_scale" not in df.columns:
        return share_cols, weighted_cols

    weights = np.where(
        df["resident_population"].astype(float) > 0,
        df["resident_population"].astype(float),
        np.where(df["population_scale"].astype(float) > 0, df["population_scale"].astype(float), 1.0),
    )
    weights_series = pd.Series(np.nan_to_num(weights, nan=1.0), index=df.index)

    for metric in SHARE_METRICS:
        if metric not in df.columns:
            continue
        metric_values = df[metric].astype(float)
        weighted_col = f"{metric}_city_weighted_mean"
        share_col = f"{metric}_share"

        weighted_sum = (metric_values * weights_series).groupby(df["month"]).transform("sum")
        weight_sum = weights_series.groupby(df["month"]).transform("sum")
        with np.errstate(divide="ignore", invalid="ignore"):
            city_weighted = weighted_sum / weight_sum.replace(0.0, np.nan)
        month_mean = metric_values.groupby(df["month"]).transform("mean")
        global_mean = metric_values.mean(skipna=True)
        city_weighted = city_weighted.fillna(month_mean).fillna(global_mean if not pd.isna(global_mean) else 0.0)
        df[weighted_col] = city_weighted.fillna(0.0)

        df[share_col] = metric_values / (df[weighted_col] + 1e-6)
        df[share_col] = df[share_col].replace([np.inf, -np.inf], 0.0).fillna(0.0)

        share_cols.append(share_col)
        weighted_cols.append(weighted_col)
        suppressible[metric].add(share_col)
        suppressible[metric].add(weighted_col)

    return share_cols, weighted_cols


def _select_search_keywords(
    df: pd.DataFrame,
    forecast_start: pd.Timestamp,
    reports_path: Path | None,
) -> Tuple[List[str], dict]:
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
    history_mask = (df["month"] < forecast_start) & df["target"].notna()
    history_df = df.loc[history_mask, ["month", "target", *search_cols]].copy()

    if history_df.empty or not search_cols:
        return [], {"folds": [], "average_mi": {}, "version": "v3.1"}

    months = sorted(history_df["month"].unique())
    fold_months = np.array_split(months, 3)
    fold_details = []
    scores = {col: [] for col in search_cols}

    for fold_idx, valid_months in enumerate(fold_months, start=1):
        if len(valid_months) == 0:
            continue
        min_valid = valid_months[0]
        train_mask = history_df["month"] < min_valid
        valid_mask = history_df["month"].isin(valid_months)
        if valid_mask.sum() == 0 or train_mask.sum() == 0:
            continue

        X_valid = history_df.loc[valid_mask, search_cols].fillna(0.0)
        y_valid = history_df.loc[valid_mask, "target"].astype(float)
        variances = X_valid.var(axis=0).to_numpy()
        constant = variances <= 1e-8
        mi_values = np.zeros_like(variances)
        if (~constant).any():
            try:
                mi_values[~constant] = mutual_info_regression(
                    X_valid.loc[:, ~constant],
                    y_valid,
                    random_state=42,
                    discrete_features=False,
                )
            except ValueError:
                mi_values[~constant] = 0.0
        fold_detail = {
            "fold": fold_idx,
            "train_month_min": str(history_df.loc[train_mask, "month"].min().date()),
            "train_month_max": str(history_df.loc[train_mask, "month"].max().date()),
            "valid_month_min": str(min(valid_months).date()),
            "valid_month_max": str(max(valid_months).date()),
            "mi": {},
        }
        for idx, col in enumerate(search_cols):
            score = float(mi_values[idx]) if not np.isnan(mi_values[idx]) else 0.0
            if constant[idx]:
                score = 0.0
            scores[col].append(score)
            fold_detail["mi"][col] = score
        fold_details.append(fold_detail)

    average_mi = {col: float(np.mean(values)) if values else 0.0 for col, values in scores.items()}
    selected = [col for col, _ in sorted(average_mi.items(), key=lambda item: item[1], reverse=True)[:12]]

    report = {
        "version": "v3.1",
        "history_month_min": str(history_df["month"].min().date()),
        "history_month_max": str(history_df["month"].max().date()),
        "folds": fold_details,
        "average_mi": average_mi,
        "selected": selected,
    }

    return selected, report


def _create_search_features(
    df: pd.DataFrame,
    selected: List[str],
    suppressible: Dict[str, set[str]],
) -> List[str]:
    created: List[str] = []
    for col in selected:
        lag_col = f"{col}_lag_1"
        rolling_col = f"{col}_rolling_mean_3"
        pct_col = f"{col}_pct_change_1"
        z_col = f"{col}_zscore_3m"

        lag_series = df.groupby("sector_id")[col].shift(1)
        df[lag_col] = lag_series.fillna(0.0)

        rolling_mean = df.groupby("sector_id")[col].transform(
            lambda s: s.shift(1).rolling(window=3, min_periods=1).mean()
        )
        df[rolling_col] = rolling_mean.fillna(0.0)

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
        suppressible[col].update({rolling_col, pct_col, z_col})
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
    if drop_cols:
        df.drop(columns=drop_cols + drop_flags, inplace=True, errors="ignore")


def _create_poi_pca(
    df: pd.DataFrame,
    history_mask: pd.Series,
) -> Tuple[List[str], dict]:
    poi_columns = [
        col
        for col in df.columns
        if (col.endswith("_dense") or col.startswith("number_of_"))
        and col not in POI_CORE_COLUMNS
    ]
    if not poi_columns:
        df["poi_pca_1"] = 0.0
        df["poi_pca_2"] = 0.0
        return ["poi_pca_1", "poi_pca_2"], {
            "original_feature_count": 0,
            "explained_variance_ratio": [0.0, 0.0],
            "columns": [],
            "used_pca": False,
        }

    training = df.loc[history_mask, poi_columns].fillna(0.0).astype(float)
    training = np.log1p(np.clip(training, 0.0, None))
    transform_all = df[poi_columns].fillna(0.0).astype(float)
    transform_all = np.log1p(np.clip(transform_all, 0.0, None))

    if training.shape[1] < 2:
        comp1 = transform_all.iloc[:, 0] if training.shape[1] == 1 else pd.Series(0.0, index=df.index)
        df["poi_pca_1"] = comp1
        df["poi_pca_2"] = 0.0
        df.drop(columns=poi_columns, inplace=True, errors="ignore")
        return ["poi_pca_1", "poi_pca_2"], {
            "original_feature_count": len(poi_columns),
            "explained_variance_ratio": [1.0, 0.0],
            "columns": poi_columns,
            "used_pca": False,
        }

    pca = PCA(n_components=2, random_state=42)
    pca.fit(training)
    transformed = pca.transform(transform_all)
    df["poi_pca_1"] = transformed[:, 0]
    df["poi_pca_2"] = transformed[:, 1]
    df.drop(columns=poi_columns, inplace=True, errors="ignore")

    summary = {
        "original_feature_count": len(poi_columns),
        "explained_variance_ratio": list(map(float, pca.explained_variance_ratio_)),
        "columns": poi_columns,
        "used_pca": True,
    }
    return ["poi_pca_1", "poi_pca_2"], summary


def _create_log1p_features(
    df: pd.DataFrame,
    log_sources: Iterable[str],
    suppressible: Dict[str, set[str]],
) -> List[str]:
    created: List[str] = []
    for column in log_sources:
        if column not in df.columns:
            continue
        log_col = f"{column}_log1p"
        if log_col in df.columns:
            continue
        df[log_col] = np.log1p(np.clip(df[column].astype(float), 0.0, None))
        created.append(log_col)
        base = _resolve_base_column(column)
        suppressible[base].add(log_col)
    return created


def _enforce_projection_guard(
    df: pd.DataFrame,
    suppressible: Dict[str, set[str]],
    drift_report: dict,
    original_df: pd.DataFrame,
    forecast_start: pd.Timestamp,
) -> dict:
    projection_metrics = drift_report.get("metrics", {}) if isinstance(drift_report, dict) else {}
    historical = _compute_historical_projection_distribution(original_df, forecast_start)
    suppressed_records = []

    for column, metrics in projection_metrics.items():
        p80_values = historical.get(column, np.array([]))
        p80_threshold = float(np.percentile(p80_values, 80)) if p80_values.size else np.inf
        mape = float(metrics.get("mape", 0.0))
        proj_ratio = float(metrics.get("proj_source_3_ratio", 0.0)) + float(
            metrics.get("proj_source_4_ratio", 0.0)
        )
        if mape > p80_threshold or proj_ratio > 0.85:
            features_to_drop = suppressible.get(column, set())
            if not features_to_drop:
                continue
            drop_list = [feature for feature in features_to_drop if feature in df.columns]
            if drop_list:
                df.drop(columns=drop_list, inplace=True, errors="ignore")
                suppressed_records.append(
                    {
                        "column": column,
                        "trigger_mape": mape,
                        "historical_mape_p80": p80_threshold,
                        "proj_source_ge3_ratio": proj_ratio,
                        "dropped_features": drop_list,
                    }
                )

    return {
        "projection_suppressed_features": suppressed_records,
        "status": "warning" if suppressed_records else "ok",
    }


def _enforce_feature_budget(
    df: pd.DataFrame,
    categories: Dict[str, List[str]],
    log_features: List[str],
) -> List[str]:
    """Trim excess log1p features to satisfy the feature count budget."""

    trimmed: List[str] = []
    feature_columns = [col for col in df.columns if col not in {"month", "id"}]
    total = len(feature_columns)
    if total <= FEATURE_COUNT_MAX:
        return trimmed

    logger.warning(
        "feature count %s exceeds budget %s, trimming log1p features",
        total,
        FEATURE_COUNT_MAX,
    )

    priority_patterns = [
        "_rolling_std_6_log1p",
        "_rolling_std_3_log1p",
        "_lag_12_log1p",
        "_rolling_mean_12_log1p",
        "_lag_6_log1p",
        "_rolling_mean_6_log1p",
        "_lag_3_log1p",
        "_rolling_mean_3_log1p",
    ]
    secondary_patterns = [
        "_lag_1_log1p",
        "_share_log1p",
        "_city_weighted_mean_log1p",
        "_growth_3_log1p",
        "_growth_1_log1p",
    ]

    def _trim_by_patterns(patterns: List[str]) -> None:
        nonlocal total
        for pattern in patterns:
            if total <= FEATURE_COUNT_MAX:
                return
            for column in list(log_features):
                if total <= FEATURE_COUNT_MAX:
                    break
                if column.endswith(pattern) and column in df.columns:
                    df.drop(columns=column, inplace=True)
                    trimmed.append(column)
                    log_features.remove(column)
                    total -= 1

    _trim_by_patterns(priority_patterns)
    if total > FEATURE_COUNT_MAX:
        _trim_by_patterns(secondary_patterns)

    if total > FEATURE_COUNT_MAX:
        for column in list(log_features):
            if total <= FEATURE_COUNT_MAX:
                break
            if column in df.columns:
                df.drop(columns=column, inplace=True)
                trimmed.append(column)
                log_features.remove(column)
                total -= 1

    return trimmed


def _compute_historical_projection_distribution(
    panel_df: pd.DataFrame,
    forecast_start: pd.Timestamp,
) -> Dict[str, np.ndarray]:
    columns = [col for col in _collect_projection_columns(panel_df) if col in panel_df.columns]
    stats = {col: [] for col in columns}
    start_month = pd.Timestamp("2019-02-01")
    cutoff = min(forecast_start, pd.Timestamp("2024-07-01"))
    base_cols = ["month", "sector_id", "city_id", *columns]
    subset = panel_df[base_cols].copy().sort_values(["month", "sector_id"])
    months = [m for m in sorted(subset["month"].unique()) if start_month <= m <= cutoff]

    for column in columns:
        sector_history: Dict[int, List[float]] = defaultdict(list)
        city_month_values: Dict[tuple[int, int], List[float]] = defaultdict(list)
        global_history: List[float] = []

        for month in months:
            month_mask = subset["month"] == month
            if not month_mask.any():
                continue
            month_rows = subset.loc[month_mask, ["sector_id", "city_id", column]]
            month_num = int(month.month)
            global_median = float(np.median(global_history)) if global_history else np.nan

            for _, row in month_rows.iterrows():
                value = row[column]
                sector_id = int(row["sector_id"])
                city_id = int(row["city_id"])
                history_list = sector_history[sector_id]
                city_key = (city_id, month_num)
                city_values = city_month_values.get(city_key, [])
                city_median_dict = {city_key: float(np.median(city_values))} if city_values else {}
                city_count_dict = {city_key: len(city_values)} if city_values else {}

                predicted, _ = _project_value(
                    history_list,
                    month,
                    city_id,
                    city_median_dict,
                    city_count_dict,
                    global_median,
                )

                if not pd.isna(value):
                    actual_val = float(value)
                    error = abs(predicted - actual_val) / max(abs(actual_val), 1.0)
                    stats[column].append(error)

            for _, row in month_rows.iterrows():
                value = row[column]
                if pd.isna(value):
                    continue
                sector_id = int(row["sector_id"])
                city_id = int(row["city_id"])
                city_key = (city_id, month_num)
                sector_history[sector_id].append(float(value))
                global_history.append(float(value))
                city_month_values[city_key].append(float(value))

    return {col: np.array(values) for col, values in stats.items()}


def _collect_projection_columns(df: pd.DataFrame) -> List[str]:
    columns: List[str] = []
    for name in PROJECTION_COLUMNS:
        if name in df.columns:
            columns.append(name)
    search = [
        col
        for col in df.columns
        if col.startswith("search_kw_")
        and "_lag_" not in col
        and "_rolling_" not in col
        and "_pct_change" not in col
        and "_zscore" not in col
        and not col.endswith("_was_missing")
    ]
    columns.extend(search)
    return columns


def _build_inventory(
    df: pd.DataFrame,
    categories: Dict[str, List[str]],
    projection_guard: dict,
    trimmed_log1p: List[str],
) -> dict:
    filtered_categories = {
        key: sorted({col for col in values if col in df.columns}) for key, values in categories.items()
    }
    feature_columns = [col for col in df.columns if col not in {"month", "id"}]
    inventory = {
        "total_features": len(feature_columns),
        "min_features": FEATURE_COUNT_MIN,
        "max_features": FEATURE_COUNT_MAX,
        "delta_vs_v30": 0,
        "by_category": filtered_categories,
        "projection_guard": projection_guard,
        "log1p_trimmed": sorted(trimmed_log1p),
    }
    return inventory


def _collect_log_sources(categories: Dict[str, List[str]]) -> List[str]:
    log_sources: set[str] = set()
    for column in METRIC_SET_FULL:
        log_sources.add(column)
    for bucket in (
        "lag",
        "lag_long",
        "rolling_mean",
        "rolling_mean_long",
        "rolling_std",
        "growth",
        "share",
        "weighted_mean",
    ):
        log_sources.update(categories.get(bucket, []))
    return sorted(log_sources)


def _resolve_base_column(name: str) -> str:
    suffixes = [
        "_lag_12",
        "_lag_6",
        "_lag_3",
        "_lag_1",
        "_rolling_mean_12",
        "_rolling_mean_6",
        "_rolling_mean_3",
        "_rolling_std_6",
        "_rolling_std_3",
        "_growth_3",
        "_growth_1",
        "_share",
        "_city_weighted_mean",
        "_pct_change_1",
        "_zscore_3m",
        "_log1p",
    ]
    for suffix in suffixes:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name
