from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

__all__ = ["build_feature_matrix_v2"]

LAG_STEPS = (1, 3, 6, 12)
ROLLING_WINDOWS = (3, 6, 12)


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    return df


def _add_missing_flags(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for col in columns:
        flag = f"{col}_was_missing"
        if flag not in df.columns:
            df[flag] = df[col].isna().astype(np.int8)
    return df


def _apply_missing_value_policy(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    report: Dict[str, Dict[str, int]] = {}
    df = df.sort_values(["sector_id", "month"]).reset_index(drop=True)

    count_columns = [
        col
        for col in df.columns
        if col.startswith(("num_", "area_", "amount_", "construction_", "transaction_", "planned_building_"))
        or col.startswith("number_of_")
        or col.endswith("_dense")
    ]
    if "target_filled" in df.columns:
        count_columns = [col for col in count_columns if col != "target_filled"]

    df = _add_missing_flags(df, count_columns)
    df, count_report = _fill_count_like_columns(df, count_columns)
    report.update(count_report)

    if "target_filled" in df.columns:
        flag_col = "target_filled_was_missing"
        if flag_col not in df.columns:
            df[flag_col] = df["target_filled"].isna().astype(np.int8)
        missing_count = int(df["target_filled"].isna().sum())
        df["target_filled"] = df["target_filled"].fillna(0.0)
        report["target_filled"] = {
            "strategy": "fill_zero",
            "filled": missing_count,
        }

    price_columns = [
        col for col in df.columns if "price" in col and not col.startswith("city_")
    ]
    df = _add_missing_flags(df, price_columns)
    df, price_report = _fill_price_columns(df, price_columns)
    report.update(price_report)

    period_columns = [col for col in df.columns if col.startswith("period_")]
    df = _add_missing_flags(df, period_columns)
    df, period_report = _fill_period_columns(df, period_columns)
    report.update(period_report)

    city_columns = [col for col in df.columns if col.startswith("city_") and not col.endswith("_was_interpolated")]
    df = _add_missing_flags(df, city_columns)
    df, city_report = _fill_city_columns(df, city_columns)
    report.update(city_report)

    return df, report


def _fill_count_like_columns(df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    if not columns:
        return df, {}
    report: Dict[str, Dict[str, int]] = {}
    df = df.sort_values(["sector_id", "month"]).reset_index(drop=True)

    for col in columns:
        filled = {
            "strategy": "sector_ffill_then_rolling_then_city_then_global_then_zero",
            "filled_sector_ffill": 0,
            "filled_sector_rolling": 0,
            "filled_city_median": 0,
            "filled_global_median": 0,
            "filled_zero": 0,
        }
        na_mask = df[col].isna()
        if not na_mask.any():
            report[col] = filled
            continue

        before_ffill = df[col].isna().sum()
        df[col] = df.groupby("sector_id")[col].ffill()
        after_ffill = df[col].isna().sum()
        filled["filled_sector_ffill"] = int(before_ffill - after_ffill)

        na_mask = df[col].isna()
        if na_mask.any():
            rolling6 = (
                df.groupby("sector_id")[col]
                .transform(lambda s: s.shift(1).rolling(window=6, min_periods=4).median())
            )
            rolling3 = (
                df.groupby("sector_id")[col]
                .transform(lambda s: s.shift(1).rolling(window=3, min_periods=2).median())
            )
            before_roll = df[col].isna().sum()
            df.loc[na_mask, col] = rolling6.loc[na_mask]
            na_mask = df[col].isna()
            df.loc[na_mask, col] = rolling3.loc[na_mask]
            after_roll = df[col].isna().sum()
            filled["filled_sector_rolling"] = int(before_roll - after_roll)

        na_mask = df[col].isna()
        if na_mask.any():
            city_median = df.groupby("month")[col].transform("median")
            before_city = na_mask.sum()
            df.loc[na_mask, col] = city_median.loc[na_mask]
            after_city = df[col].isna().sum()
            filled["filled_city_median"] = int(before_city - after_city)

        na_mask = df[col].isna()
        if na_mask.any():
            global_median = float(df[col].median(skipna=True)) if not df[col].dropna().empty else 0.0
            before_global = na_mask.sum()
            df.loc[na_mask, col] = global_median
            after_global = df[col].isna().sum()
            filled["filled_global_median"] = int(before_global - after_global)

        na_mask = df[col].isna()
        if na_mask.any():
            df.loc[na_mask, col] = 0.0
            filled["filled_zero"] = int(na_mask.sum())

        report[col] = filled
    return df, report


def _fill_price_columns(df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    if not columns:
        return df, {}
    report: Dict[str, Dict[str, int]] = {}
    df = df.sort_values(["sector_id", "month"]).reset_index(drop=True)

    for col in columns:
        filled = {
            "strategy": "sector_ffill_then_history_median_then_global",
            "filled_sector_ffill": 0,
            "filled_sector_history": 0,
            "filled_global_median": 0,
        }
        df[col] = pd.to_numeric(df[col], errors="coerce")
        before_ffill = df[col].isna().sum()
        df[col] = df.groupby("sector_id")[col].ffill()
        after_ffill = df[col].isna().sum()
        filled["filled_sector_ffill"] = int(before_ffill - after_ffill)

        na_mask = df[col].isna()
        if na_mask.any():
            history = df.groupby("sector_id")[col].transform(
                lambda s: _sector_history_fill(s)
            )
            before_hist = na_mask.sum()
            df.loc[na_mask, col] = history.loc[na_mask]
            after_hist = df[col].isna().sum()
            filled["filled_sector_history"] = int(before_hist - after_hist)

        na_mask = df[col].isna()
        if na_mask.any():
            global_median = float(df[col].median(skipna=True)) if not df[col].dropna().empty else 0.0
            before_global = na_mask.sum()
            df.loc[na_mask, col] = global_median
            after_global = df[col].isna().sum()
            filled["filled_global_median"] = int(before_global - after_global)

        df[col] = df[col].fillna(0.0)
        report[col] = filled

    return df, report


def _sector_history_fill(series: pd.Series) -> pd.Series:
    history_values: List[float] = []
    filled_values: List[float] = []
    for value in series:
        filled_values.append(float(np.nanmedian(history_values)) if history_values else np.nan)
        if not pd.isna(value):
            history_values.append(value)
    result = pd.Series(filled_values, index=series.index)
    remaining_mask = series.isna() & result.isna()
    if remaining_mask.any():
        global_median = float(series.median(skipna=True)) if not pd.isna(series.median(skipna=True)) else 0.0
        result.loc[remaining_mask] = global_median
    return result


def _fill_period_columns(df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    if not columns:
        return df, {}
    report: Dict[str, Dict[str, int]] = {}
    df = df.sort_values(["sector_id", "month"]).reset_index(drop=True)

    for col in columns:
        filled = {
            "strategy": "sector_ffill_then_sector_median_then_global",
            "filled_sector_ffill": 0,
            "filled_sector_median": 0,
            "filled_global_median": 0,
        }
        before_ffill = df[col].isna().sum()
        df[col] = df.groupby("sector_id")[col].ffill()
        after_ffill = df[col].isna().sum()
        filled["filled_sector_ffill"] = int(before_ffill - after_ffill)

        na_mask = df[col].isna()
        if na_mask.any():
            sector_median = df.groupby("sector_id")[col].transform(lambda s: s.median(skipna=True))
            before_median = na_mask.sum()
            df.loc[na_mask, col] = sector_median.loc[na_mask]
            after_median = df[col].isna().sum()
            filled["filled_sector_median"] = int(before_median - after_median)

        na_mask = df[col].isna()
        if na_mask.any():
            global_median = float(df[col].median(skipna=True)) if not df[col].dropna().empty else 0.0
            before_global = na_mask.sum()
            df.loc[na_mask, col] = global_median
            after_global = df[col].isna().sum()
            filled["filled_global_median"] = int(before_global - after_global)

        df[col] = df[col].fillna(0.0)
        report[col] = filled
    return df, report


def _fill_city_columns(df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    if not columns:
        return df, {}
    report: Dict[str, Dict[str, int]] = {}
    df = df.sort_values(["month", "sector_id"]).reset_index(drop=True)

    for col in columns:
        filled = {
            "strategy": "sector_ffill_then_month_ffill_then_global_median",
            "filled_sector_ffill": 0,
            "filled_month_ffill": 0,
            "filled_global_median": 0,
        }
        before_sector = df[col].isna().sum()
        df[col] = df.groupby("sector_id")[col].ffill()
        after_sector = df[col].isna().sum()
        filled["filled_sector_ffill"] = int(before_sector - after_sector)

        na_mask = df[col].isna()
        if na_mask.any():
            before_month = na_mask.sum()
            df[col] = df[col].ffill()
            after_month = df[col].isna().sum()
            filled["filled_month_ffill"] = int(before_month - after_month)

        na_mask = df[col].isna()
        if na_mask.any():
            global_median = float(df[col].median(skipna=True)) if not df[col].dropna().empty else 0.0
            before_global = na_mask.sum()
            df.loc[na_mask, col] = global_median
            after_global = df[col].isna().sum()
            filled["filled_global_median"] = int(before_global - after_global)

        df[col] = df[col].fillna(0.0)
        report[col] = filled
    return df, report


def _compute_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["month"].dt.year
    df["month_num"] = df["month"].dt.month
    df["quarter"] = df["month"].dt.quarter
    df["month_index"] = (df["year"] - 2019) * 12 + (df["month_num"] - 1)
    df["is_year_start"] = (df["month_num"] == 1).astype(int)
    df["is_year_end"] = (df["month_num"] == 12).astype(int)
    df["days_in_month"] = df["month"].dt.days_in_month
    return df


def _compute_lag_and_rolling(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    metadata: Dict[str, List[str]] = {
        "lag_features": [],
        "rolling_mean_features": [],
        "rolling_std_features": [],
        "rolling_min_features": [],
        "rolling_max_features": [],
        "rolling_count_features": [],
        "growth_features": [],
    }
    value_cols = [
        col
        for col in df.columns
        if col.startswith(("num_", "area_", "amount_", "price_", "total_price_")) or col == "target_filled"
    ]

    df.sort_values(["sector_id", "month"], inplace=True)
    grouped = df.groupby("sector_id", sort=False)

    for col in value_cols:
        for lag in LAG_STEPS:
            lag_col = f"{col}_lag_{lag}"
            df[lag_col] = grouped[col].shift(lag)
            metadata["lag_features"].append(lag_col)
        for window in ROLLING_WINDOWS:
            shifted = grouped[col].shift(1)
            mean_col = f"{col}_rolling_mean_{window}"
            std_col = f"{col}_rolling_std_{window}"
            min_col = f"{col}_rolling_min_{window}"
            max_col = f"{col}_rolling_max_{window}"
            count_col = f"{col}_rolling_count_{window}"

            df[mean_col] = shifted.groupby(df["sector_id"]).rolling(window=window, min_periods=1).mean().reset_index(level=0, drop=True)
            df[std_col] = shifted.groupby(df["sector_id"]).rolling(window=window, min_periods=1).std(ddof=0).reset_index(level=0, drop=True)
            df[min_col] = shifted.groupby(df["sector_id"]).rolling(window=window, min_periods=1).min().reset_index(level=0, drop=True)
            df[max_col] = shifted.groupby(df["sector_id"]).rolling(window=window, min_periods=1).max().reset_index(level=0, drop=True)
            df[count_col] = shifted.groupby(df["sector_id"]).rolling(window=window, min_periods=1).count().reset_index(level=0, drop=True)

            metadata["rolling_mean_features"].append(mean_col)
            metadata["rolling_std_features"].append(std_col)
            metadata["rolling_min_features"].append(min_col)
            metadata["rolling_max_features"].append(max_col)
            metadata["rolling_count_features"].append(count_col)

        lag1 = df[f"{col}_lag_1"]
        lag3 = df[f"{col}_lag_3"]
        growth_1 = f"{col}_growth_1"
        growth_3 = f"{col}_growth_3"
        df[growth_1] = np.where(lag1.abs() > 0, (df[col] - lag1) / (lag1.abs() + 1e-6), 0.0)
        df[growth_3] = np.where(lag3.abs() > 0, (df[col] - lag3) / (lag3.abs() + 1e-6), 0.0)
        df.loc[lag1.isna(), growth_1] = 0.0
        df.loc[lag3.isna(), growth_3] = 0.0
        df[growth_1 + "_was_missing"] = lag1.isna().astype(np.int8)
        df[growth_3 + "_was_missing"] = lag3.isna().astype(np.int8)
        metadata["growth_features"].extend([growth_1, growth_3])

    return df, metadata


def _population_weighted_features(df: pd.DataFrame, metrics: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    share_features: List[str] = []
    weights = df["resident_population"].replace({0: np.nan})
    weights = weights.fillna(df.get("population_scale", 1.0))
    weights = weights.fillna(1.0)

    for col in metrics:
        weighted_sum = (df[col] * weights).groupby(df["month"]).transform("sum")
        weight_total = weights.groupby(df["month"]).transform("sum")
        simple_mean = df.groupby("month")[col].transform("mean")
        mean_col = f"{col}_city_weighted_mean"
        df[mean_col] = np.where(weight_total != 0, weighted_sum / weight_total, simple_mean)
        share_col = f"{col}_share"
        df[share_col] = df[col] / (df[mean_col] + 1e-6)
        share_features.append(share_col)

    return df, share_features


def _interaction_terms(df: pd.DataFrame) -> List[str]:
    interaction_cols: List[str] = []
    for col in df.columns:
        if col.endswith("_nearby"):
            base = col[: -len("_nearby")]
            if base in df.columns:
                diff_col = f"{base}_vs_nearby_diff"
                df[diff_col] = df[base] - df[col]
                interaction_cols.append(diff_col)
    return interaction_cols


def _search_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    search_cols = [col for col in df.columns if col.startswith("search_kw_") and not col.endswith("_was_missing")]
    derived_cols: List[str] = []
    if not search_cols:
        return df, derived_cols

    df = df.sort_values(["sector_id", "month"]).reset_index(drop=True)
    grouped = df.groupby("sector_id", sort=False)
    zscore_cols = []

    for col in search_cols:
        pct_col = f"{col}_pct_change"
        z_col = f"{col}_zscore_3m"
        prev = grouped[col].shift(1)
        df[pct_col] = np.where(prev.abs() > 0, (df[col] - prev) / (prev.abs() + 1e-6), 0.0)
        mean = prev.groupby(df["sector_id"]).rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
        std = prev.groupby(df["sector_id"]).rolling(window=3, min_periods=1).std(ddof=0).reset_index(level=0, drop=True)
        df[z_col] = np.where(std > 0, (prev - mean) / std, 0.0)
        df[z_col] = df[z_col].fillna(0.0)
        derived_cols.extend([pct_col, z_col])
        zscore_cols.append(z_col)

    if zscore_cols:
        mean_z = df[zscore_cols].mean(axis=1)
        density_cols = [col for col in df.columns if col.endswith("_dense")]
        for col in density_cols:
            adj_col = f"{col}_density_adj"
            df[adj_col] = df[col] * mean_z
            derived_cols.append(adj_col)
    return df, derived_cols


def _apply_log_transforms(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    log_cols = [
        col
        for col in df.columns
        if col.startswith(("num_", "area_", "amount_", "number_of_"))
        or col.endswith("_dense")
        or col == "target_filled"
    ]
    transformed: List[str] = []
    for col in log_cols:
        log_col = f"{col}_log1p"
        df[log_col] = np.log1p(df[col])
        transformed.append(log_col)
    return df, transformed


def _run_poi_pca(df: pd.DataFrame, log_density_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    from sklearn.decomposition import PCA

    if len(log_density_cols) < 2:
        return df, []
    matrix = df[log_density_cols].fillna(0.0)
    if matrix.var().sum() == 0:
        return df, []
    pca = PCA()
    pca.fit(matrix)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumulative, 0.95) + 1)
    n_components = min(n_components, matrix.shape[1])
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(matrix)
    cols: List[str] = []
    for idx in range(components.shape[1]):
        name = f"poi_pca_{idx + 1}"
        df[name] = components[:, idx]
        cols.append(name)
    return df, cols


def build_feature_matrix_v2(
    panel_df: pd.DataFrame,
    features_path: str | Path | None = None,
    reports_dir: str | Path | None = None,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    df = _ensure_datetime(panel_df)
    df = df.sort_values(["sector_id", "month"]).reset_index(drop=True)

    df, missing_report = _apply_missing_value_policy(df)

    df = _compute_time_features(df)

    metric_base_cols = [
        col
        for col in df.columns
        if col.startswith(("num_", "area_", "amount_", "price_", "total_price_")) or col == "target_filled"
    ]
    df, lag_metadata = _compute_lag_and_rolling(df)
    df, share_features = _population_weighted_features(df, metric_base_cols)
    interaction_features = _interaction_terms(df)
    df, search_features = _search_features(df)
    df, log_features = _apply_log_transforms(df)
    density_log_cols = [col for col in log_features if col.endswith("_dense_log1p")]
    df, poi_pca_cols = _run_poi_pca(df, density_log_cols)

    metadata: Dict[str, List[str]] = {
        **lag_metadata,
        "share_features": share_features,
        "interaction_features": interaction_features,
        "search_features": search_features,
        "log1p_features": log_features,
        "poi_pca_features": poi_pca_cols,
    }

    if reports_dir is not None:
        reports_dir = Path(reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / "missing_value_report.json"
        with report_path.open("w", encoding="utf-8") as fp:
            json.dump(missing_report, fp, indent=2)

        metadata_path = reports_dir / "feature_metadata.json"
        with metadata_path.open("w", encoding="utf-8") as fp:
            json.dump(metadata, fp, indent=2)

    if features_path is not None:
        features_path = Path(features_path)
        if features_path.exists():
            if features_path.is_dir():
                for item in features_path.glob("**/*"):
                    if item.is_file():
                        item.unlink()
                for item in sorted(features_path.glob("*"), reverse=True):
                    if item.is_dir():
                        item.rmdir()
            else:
                features_path.unlink()
        df.assign(year=df["month"].dt.year).to_parquet(features_path, index=False, partition_cols=["year"])

    return df, metadata
