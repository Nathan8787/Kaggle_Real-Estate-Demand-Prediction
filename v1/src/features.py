from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


LAG_STEPS = (1, 3, 6, 12)
ROLLING_WINDOWS = (3, 6, 12)


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['month'] = pd.to_datetime(df['month'])
    return df


def _add_missing_indicators(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            continue
        indicator_col = f"{col}_was_missing"
        if indicator_col in df.columns:
            continue
        df[indicator_col] = df[col].isna().astype(np.int8)
    return df


def _fill_count_features(
    df: pd.DataFrame,
    columns: Iterable[str],
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    report: Dict[str, Dict[str, int]] = {}
    columns = [col for col in columns if col in df.columns]
    if not columns:
        return df, report
    df = df.sort_values(['sector_id', 'month']).reset_index(drop=True)
    for col in columns:
        initial_mask = df[col].isna()
        if not initial_mask.any():
            report[col] = {
                'strategy': 'sector_ffill_then_median_then_global_else_zero',
                'filled_ffill': 0,
                'filled_sector_median': 0,
                'filled_global_median': 0,
                'filled_zero': 0,
            }
            continue
        df[col] = pd.to_numeric(df[col], errors='coerce')
        initial_missing_total = int(initial_mask.sum())
        df[col] = df.groupby('sector_id')[col].ffill()
        after_ffill_mask = df[col].isna()
        filled_ffill = int(initial_missing_total - after_ffill_mask.sum())
        sector_mask = after_ffill_mask.copy()
        filled_sector = 0
        if sector_mask.any():
            sector_medians = df.groupby('sector_id')[col].transform('median')
            df.loc[sector_mask, col] = sector_medians.loc[sector_mask]
            after_sector_mask = df[col].isna()
            filled_sector = int(sector_mask.sum() - after_sector_mask.sum())
        else:
            after_sector_mask = sector_mask
        global_mask = after_sector_mask.copy()
        filled_global = 0
        if global_mask.any():
            global_median = float(df[col].median(skipna=True))
            if np.isnan(global_median):
                global_median = 0.0
            df.loc[global_mask, col] = global_median
            filled_global = int(global_mask.sum())
        remaining_mask = df[col].isna()
        filled_zero = int(remaining_mask.sum())
        if filled_zero:
            df.loc[remaining_mask, col] = 0.0
        report[col] = {
            'strategy': 'sector_ffill_then_median_then_global_else_zero',
            'filled_ffill': filled_ffill,
            'filled_sector_median': filled_sector,
            'filled_global_median': filled_global,
            'filled_zero': filled_zero,
        }
    return df, report


def _apply_missing_value_policy(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    report: Dict[str, Dict[str, int]] = {}
    df = df.sort_values(['sector_id', 'month']).reset_index(drop=True)

    count_cols = [
        col
        for col in df.columns
        if (
            col.startswith(('num_', 'area_', 'amount_', 'construction_', 'transaction_', 'planned_building_'))
            or 'number_of_' in col
            or col.endswith('_dense')
        )
    ]
    count_cols = [col for col in count_cols if col in df.columns and col != 'target_filled']
    if count_cols:
        df = _add_missing_indicators(df, count_cols)
        df, count_report = _fill_count_features(df, count_cols)
        report.update(count_report)

    if 'target_filled' in df.columns:
        indicator_col = 'target_filled_was_missing'
        if indicator_col not in df.columns:
            df[indicator_col] = df['target_filled'].isna().astype(np.int8)
        target_missing = int(df['target_filled'].isna().sum())
        if target_missing:
            df['target_filled'] = df['target_filled'].fillna(0.0)
        report['target_filled'] = {'strategy': 'fill_zero', 'filled': target_missing}

    price_cols = [
        col
        for col in df.columns
        if 'price' in col and not col.startswith('city_')
    ]
    if price_cols:
        df = _add_missing_indicators(df, price_cols)
        df, price_report = _fill_with_sector_history(
            df,
            price_cols,
            strategy_name='sector_ffill_then_hist_median_then_global',
        )
        report.update(price_report)

    period_cols = [col for col in df.columns if col.startswith('period_')]
    if period_cols:
        df = _add_missing_indicators(df, period_cols)
        df, period_report = _fill_period_features(df, period_cols)
        report.update(period_report)

    city_cols = [col for col in df.columns if col.startswith('city_')]
    if city_cols:
        df = _add_missing_indicators(df, city_cols)
        df = df.sort_values(['month', 'sector_id']).reset_index(drop=True)
        for col in city_cols:
            original_mask = df[col].isna()
            filled_info = {
                'strategy': 'sector_ffill_then_global_ffill_then_global_median',
                'filled_sector_ffill': 0,
                'filled_global_ffill': 0,
                'filled_global_median': 0,
            }
            if not original_mask.any():
                report[col] = filled_info
                continue
            df[col] = df.groupby('sector_id')[col].ffill()
            after_sector_mask = df[col].isna()
            filled_info['filled_sector_ffill'] = int(original_mask.sum() - after_sector_mask.sum())
            df[col] = df[col].ffill()
            after_global_mask = df[col].isna()
            filled_info['filled_global_ffill'] = int(after_sector_mask.sum() - after_global_mask.sum())
            if after_global_mask.any():
                global_median = float(df[col].median(skipna=True))
                if np.isnan(global_median):
                    global_median = 0.0
                df.loc[after_global_mask, col] = global_median
                filled_info['filled_global_median'] = int(after_global_mask.sum())
            report[col] = filled_info

    return df, report


def _fill_with_sector_history(
    df: pd.DataFrame,
    columns: List[str],
    strategy_name: str,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    if not columns:
        return df, {}
    report: Dict[str, Dict[str, int]] = {}
    df = df.sort_values(['sector_id', 'month']).reset_index(drop=True)
    for col in columns:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors='coerce')
        initial_mask = df[col].isna()
        filled_record = {
            'strategy': strategy_name,
            'filled_ffill': 0,
            'filled_sector_median': 0,
            'filled_global_median': 0,
        }
        if not initial_mask.any():
            report[col] = filled_record
            continue
        df[col] = df.groupby('sector_id')[col].ffill()
        after_ffill_mask = df[col].isna()
        filled_record['filled_ffill'] = int(initial_mask.sum() - after_ffill_mask.sum())
        filled_sector = 0
        for sector, sector_frame in df.groupby('sector_id', sort=False):
            idx = sector_frame.index
            values = sector_frame[col]
            if not values.isna().any():
                continue
            history: List[float] = []
            fill_values: List[float] = []
            for value in values:
                fill_values.append(float(np.nanmedian(history)) if history else np.nan)
                if not pd.isna(value):
                    history.append(value)
            fill_series = pd.Series(fill_values, index=idx)
            mask = df.loc[idx, col].isna()
            df.loc[idx, col] = df.loc[idx, col].where(~mask, fill_series)
            filled_sector += int(mask.sum())
        filled_record['filled_sector_median'] = filled_sector
        remaining_mask = df[col].isna()
        if remaining_mask.any():
            global_median = float(df[col].median(skipna=True))
            if np.isnan(global_median):
                global_median = 0.0
            fill_count = int(remaining_mask.sum())
            df.loc[remaining_mask, col] = global_median
            filled_record['filled_global_median'] = fill_count
        report[col] = filled_record
    return df, report


def _fill_period_features(
    df: pd.DataFrame,
    columns: List[str],
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    if not columns:
        return df, {}
    report: Dict[str, Dict[str, int]] = {}
    df = df.sort_values(['sector_id', 'month']).reset_index(drop=True)
    for col in columns:
        original_na = df[col].isna().sum()
        if original_na == 0:
            report[col] = {"strategy": "forward_fill_sector_then_median", "filled": 0}
            continue
        for sector, sector_frame in df.groupby('sector_id', sort=False):
            idx = sector_frame.index
            ordered = sector_frame.sort_values('month')
            df.loc[ordered.index, col] = ordered[col].ffill()
        sector_na = df[col].isna()
        if sector_na.any():
            df, sector_report = _fill_with_sector_history(df, [col], strategy_name="sector_median")
            report.update(sector_report)
        remaining = df[col].isna().sum()
        if remaining:
            global_median = float(df[col].median(skipna=True)) if not pd.isna(df[col].median(skipna=True)) else 0.0
            df.loc[df[col].isna(), col] = global_median
        final_filled = original_na - int(df[col].isna().sum())
        report[col] = {"strategy": "forward_fill_sector_then_median", "filled": int(final_filled)}
    return df, report


def _compute_time_features(df: pd.DataFrame) -> pd.DataFrame:
    base_year = 2019
    base_month = 1
    df['year'] = df['month'].dt.year
    df['month_num'] = df['month'].dt.month
    df['quarter'] = df['month'].dt.quarter
    df['month_index'] = (df['year'] - base_year) * 12 + (df['month_num'] - base_month)
    df['is_year_start'] = (df['month_num'] == 1).astype(int)
    df['is_year_end'] = (df['month_num'] == 12).astype(int)
    df['days_in_month'] = df['month'].dt.days_in_month
    return df


def _log1p_transform(df: pd.DataFrame, columns: List[str]) -> Dict[str, str]:
    created: Dict[str, str] = {}
    for col in columns:
        safe = np.clip(df[col].astype(float), a_min=0.0, a_max=None)
        new_col = f"{col}_log1p"
        df[new_col] = np.log1p(safe)
        created[col] = new_col
    return created


def _compute_lag_and_rolling(
    df: pd.DataFrame,
    metrics: List[str],
) -> Dict[str, List[str]]:
    metadata: Dict[str, List[str]] = {
        'lag_features': [],
        'rolling_mean_features': [],
        'rolling_std_features': [],
        'rolling_min_features': [],
        'rolling_max_features': [],
        'rolling_count_features': [],
        'growth_features': [],
    }
    df = df.sort_values(['sector_id', 'month']).reset_index(drop=True)
    for col in metrics:
        for lag in LAG_STEPS:
            lag_name = f"{col}_lag_{lag}"
            df[lag_name] = df.groupby('sector_id')[col].shift(lag)
            metadata['lag_features'].append(lag_name)
        for window in ROLLING_WINDOWS:
            group = df.groupby('sector_id')[col]
            shifted = group.shift(1)
            mean_name = f"{col}_rolling_mean_{window}"
            std_name = f"{col}_rolling_std_{window}"
            min_name = f"{col}_rolling_min_{window}"
            max_name = f"{col}_rolling_max_{window}"
            count_name = f"{col}_rolling_count_{window}"

            mean_series = shifted.groupby(df['sector_id']).rolling(window=window, min_periods=1).mean().reset_index(level=0, drop=True)
            std_series = shifted.groupby(df['sector_id']).rolling(window=window, min_periods=1).std(ddof=0).reset_index(level=0, drop=True)
            min_series = shifted.groupby(df['sector_id']).rolling(window=window, min_periods=1).min().reset_index(level=0, drop=True)
            max_series = shifted.groupby(df['sector_id']).rolling(window=window, min_periods=1).max().reset_index(level=0, drop=True)
            count_series = shifted.groupby(df['sector_id']).rolling(window=window, min_periods=1).count().reset_index(level=0, drop=True)

            df[mean_name] = mean_series
            df[std_name] = std_series
            df[min_name] = min_series
            df[max_name] = max_series
            df[count_name] = count_series

            metadata['rolling_mean_features'].append(mean_name)
            metadata['rolling_std_features'].append(std_name)
            metadata['rolling_min_features'].append(min_name)
            metadata['rolling_max_features'].append(max_name)
            metadata['rolling_count_features'].append(count_name)

        lag1 = df[f"{col}_lag_1"]
        lag3 = df[f"{col}_lag_3"]
        growth_1_name = f"{col}_growth_1"
        growth_3_name = f"{col}_growth_3"
        df[growth_1_name] = (df[col] - lag1) / (lag1.abs() + 1e-6)
        df[growth_3_name] = (df[col] - lag3) / (lag3.abs() + 1e-6)
        metadata['growth_features'].extend([growth_1_name, growth_3_name])

    return metadata


def _population_weighted_features(df: pd.DataFrame, metrics: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    share_features: List[str] = []
    weights = df['resident_population'].astype(float)
    fallback = df.get('population_scale', pd.Series(1.0, index=df.index)).astype(float)
    weights = np.where(weights > 0, weights, fallback)
    weights = np.where(weights > 0, weights, 1.0)
    weight_series = pd.Series(weights, index=df.index)

    for col in metrics:
        weighted_sum = (df[col] * weight_series).groupby(df['month']).sum()
        weight_totals = weight_series.groupby(df['month']).sum()
        simple_mean = df.groupby('month')[col].mean()
        city_mean = (weighted_sum / weight_totals.replace(0, np.nan)).fillna(simple_mean)
        mean_col = f"{col}_city_weighted_mean"
        share_col = f"{col}_share"
        df[mean_col] = df['month'].map(city_mean)
        df[share_col] = df[col] / (df[mean_col] + 1e-6)
        share_features.append(share_col)
    return df, share_features


def _interaction_terms(df: pd.DataFrame) -> List[str]:
    interaction_cols: List[str] = []
    for col in [c for c in df.columns if c.endswith('_nearby')]:
        base_col = col[:-len('_nearby')]
        if base_col in df.columns:
            new_col = f"{base_col}_vs_nearby_diff"
            df[new_col] = df[base_col] - df[col]
            interaction_cols.append(new_col)
    return interaction_cols


def _search_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    search_cols = [col for col in df.columns if col.startswith('search_kw_')]
    pct_cols: List[str] = []
    zscore_cols: List[str] = []
    if not search_cols:
        return df, []
    df = df.sort_values(['sector_id', 'month']).reset_index(drop=True)
    for col in search_cols:
        pct_col = f"{col}_pct_change"
        z_col = f"{col}_zscore_3m"
        pct_cols.append(pct_col)
        zscore_cols.append(z_col)
        for sector, sector_frame in df.groupby('sector_id', sort=False):
            idx = sector_frame.index
            ordered = sector_frame.sort_values('month')
            series = ordered[col]
            shifted = series.shift(1)
            pct = (series - shifted) / (shifted.abs() + 1e-6)
            rolling = shifted.rolling(window=3, min_periods=1)
            zscore = (shifted - rolling.mean()) / rolling.std(ddof=0)
            df.loc[ordered.index, pct_col] = pct.values
            df.loc[ordered.index, z_col] = zscore.values
    df[zscore_cols] = df[zscore_cols].fillna(0.0)
    mean_z = df[zscore_cols].mean(axis=1)
    density_cols = [col for col in df.columns if col.endswith('_dense')]
    density_adj_cols: List[str] = []
    for col in density_cols:
        new_col = f"{col}_adj"
        df[new_col] = df[col] * mean_z
        density_adj_cols.append(new_col)
    return df, pct_cols + zscore_cols + density_adj_cols


def _run_pca_on_poi(df: pd.DataFrame, density_log_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    if not density_log_cols:
        return df, []
    matrix = df[density_log_cols].fillna(0.0)
    if matrix.empty:
        return df, []
    pca = PCA()
    pca.fit(matrix)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumulative, 0.95) + 1)
    pca = PCA(n_components=min(n_components, matrix.shape[1]))
    components = pca.fit_transform(matrix)
    component_cols: List[str] = []
    for i in range(components.shape[1]):
        col_name = f"poi_pca_{i + 1}"
        df[col_name] = components[:, i]
        component_cols.append(col_name)
    return df, component_cols


def build_feature_matrix(
    panel_df: pd.DataFrame,
    features_path: Path | None = None,
    reports_dir: Path | None = None,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    df = _ensure_datetime(panel_df)
    df, missing_report = _apply_missing_value_policy(df)

    if 'sector_id' not in df.columns:
        raise KeyError('sector_id column missing from panel.')

    df = df.sort_values(['sector_id', 'month']).reset_index(drop=True)
    df = _compute_time_features(df)

    metric_columns = ['target_filled'] + [
        col for col in df.columns
        if col.startswith(('num_', 'area_', 'amount_', 'price_', 'total_price_'))
    ]
    metric_columns = sorted(set(metric_columns))
    lag_metadata = _compute_lag_and_rolling(df, metric_columns)

    df, share_features = _population_weighted_features(df, metric_columns)
    interaction_features = _interaction_terms(df)
    search_related_features = []
    df, search_related_features = _search_features(df)

    count_like_cols = [
        col for col in df.columns
        if col.startswith(('num_', 'area_', 'amount_')) or col == 'target_filled' or 'number_of_' in col or col.endswith('_dense')
    ]
    log_mapping = _log1p_transform(df, count_like_cols)
    density_log_cols = [name for orig, name in log_mapping.items() if orig.endswith('_dense')]
    df, poi_pca_cols = _run_pca_on_poi(df, density_log_cols)

    metadata: Dict[str, List[str]] = {**lag_metadata}
    metadata['share_features'] = share_features
    metadata['interaction_features'] = interaction_features
    metadata['search_features'] = search_related_features
    metadata['poi_pca_features'] = poi_pca_cols
    metadata['log1p_features'] = list(log_mapping.values())

    if reports_dir:
        reports_dir = Path(reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / 'missing_value_report.json'
        with report_path.open('w', encoding='utf-8') as fp:
            json.dump(missing_report, fp, ensure_ascii=False, indent=2)

    if features_path:
        features_path = Path(features_path)
        if features_path.exists():
            if features_path.is_dir():
                for child in features_path.glob('*'):
                    if child.is_dir():
                        import shutil
                        shutil.rmtree(child)
                    else:
                        child.unlink()
            else:
                features_path.unlink()
        df.assign(year=df['month'].dt.year).to_parquet(features_path, index=False, partition_cols=['year'])

    return df, metadata
