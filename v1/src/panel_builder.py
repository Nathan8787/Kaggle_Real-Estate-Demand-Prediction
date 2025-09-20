from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


def _parse_month(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, format="%Y-%b", errors="coerce")
    if parsed.isna().any():
        fallback = pd.to_datetime(series, errors="coerce")
        parsed = parsed.fillna(fallback)
    if parsed.isna().any():
        bad = series[parsed.isna()].unique()
        raise ValueError(f"Unable to parse month values: {bad}")
    return parsed.dt.to_period("M").dt.to_timestamp()


def _extract_month_from_id(test_df: pd.DataFrame) -> pd.Series:
    month_str = test_df['id'].str.extract(r'^(\d{4} \w{3})')[0]
    return pd.to_datetime(month_str, format="%Y %b", errors="coerce").dt.to_period('M').dt.to_timestamp()


def _extract_sector_from_id(test_df: pd.DataFrame) -> pd.Series:
    return test_df['id'].str.extract(r'(sector \d+)')[0]


def build_calendar(raw_tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    new_house = raw_tables["new_house_transactions"].copy()
    new_house["month"] = _parse_month(new_house["month"])
    sectors = set(new_house["sector"].unique())

    max_train_month = new_house["month"].max()
    start_month = new_house["month"].min()
    max_test_month = None

    if "test" in raw_tables:
        test_df = raw_tables["test"].copy()
        test_months = _extract_month_from_id(test_df)
        test_sectors = _extract_sector_from_id(test_df)
        sectors.update(test_sectors.dropna().unique())
        max_test_month = test_months.max()

    end_month = max_train_month
    if max_test_month is not None and max_test_month > end_month:
        end_month = max_test_month

    months = pd.date_range(start_month, end_month, freq="MS")
    sectors = sorted(sector for sector in sectors if pd.notna(sector))
    calendar = (
        pd.MultiIndex.from_product([months, sectors], names=["month", "sector"])
        .to_frame(index=False)
        .sort_values(["month", "sector"], ignore_index=True)
    )
    return calendar


def _prepare_monthly(df: pd.DataFrame, month_col: str = "month") -> pd.DataFrame:
    out = df.copy()
    out[month_col] = _parse_month(out[month_col])
    return out


def _suffix_columns(df: pd.DataFrame, suffix: str, skip: Iterable[str]) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        if col in skip:
            continue
        base = col.replace('_nearby_sectors', '')
        rename_map[col] = f'{base}{suffix}'
    return df.rename(columns=rename_map)


def _expand_city_indexes(df: pd.DataFrame, months: pd.DatetimeIndex) -> pd.DataFrame:
    df = df.copy()
    df["city_indicator_data_year"] = df["city_indicator_data_year"].astype(int)
    records = []
    for _, row in df.iterrows():
        year = int(row["city_indicator_data_year"])
        year_months = pd.date_range(f"{year}-01-01", f"{year}-12-01", freq="MS")
        for month in year_months:
            rec = row.to_dict()
            rec["month"] = month
            records.append(rec)
    monthly = pd.DataFrame(records)
    monthly = monthly.drop(columns=["city_indicator_data_year"], errors="ignore")
    monthly = monthly.sort_values("month").drop_duplicates(subset="month", keep="last")
    monthly = monthly.set_index("month").reindex(months).ffill().reset_index()
    monthly.columns = ["month"] + [f"city_{c}" for c in monthly.columns[1:]]
    return monthly


def _build_search_features(search_df: pd.DataFrame, months: pd.DatetimeIndex) -> pd.DataFrame:
    df = _prepare_monthly(search_df)
    agg = df.groupby(["month", "keyword_slug"], as_index=False)["search_volume"].sum()
    variance = agg.groupby("keyword_slug")["search_volume"].var(ddof=0).fillna(0)
    top_slugs = variance.sort_values(ascending=False).head(30).index.tolist()
    if not top_slugs:
        top_slugs = variance.index.tolist()
    pivot = agg[agg["keyword_slug"].isin(top_slugs)].pivot(
        index="month", columns="keyword_slug", values="search_volume"
    )
    pivot = pivot.reindex(index=months, fill_value=0.0)
    pivot = pivot.reindex(columns=top_slugs, fill_value=0.0)
    pivot.columns = [f"search_kw_{slug}" for slug in top_slugs]
    pivot = pivot.reset_index().rename(columns={'index': 'month'})
    return pivot


def merge_sources(calendar_df: pd.DataFrame, raw_tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    months = pd.date_range(calendar_df["month"].min(), calendar_df["month"].max(), freq="MS")
    panel = calendar_df.copy()

    monthly_tables: Tuple[Tuple[str, pd.DataFrame, Iterable[str]], ...] = (
        ("new_house_transactions", raw_tables["new_house_transactions"], ("month", "sector")),
        ("new_house_transactions_nearby_sectors", raw_tables["new_house_transactions_nearby_sectors"], ("month", "sector")),
        ("pre_owned_house_transactions", raw_tables["pre_owned_house_transactions"], ("month", "sector")),
        ("pre_owned_house_transactions_nearby_sectors", raw_tables["pre_owned_house_transactions_nearby_sectors"], ("month", "sector")),
        ("land_transactions", raw_tables["land_transactions"], ("month", "sector")),
        ("land_transactions_nearby_sectors", raw_tables["land_transactions_nearby_sectors"], ("month", "sector")),
    )

    for name, df, keys in monthly_tables:
        prepared = _prepare_monthly(df)
        if name.endswith("nearby_sectors"):
            prepared = _suffix_columns(prepared, "_nearby", skip=keys)
        prepared = prepared.drop_duplicates(subset=list(keys)).reset_index(drop=True)
        panel = panel.merge(prepared, on=list(keys), how="left")

    sector_poi = raw_tables["sector_poi"].copy()
    panel = panel.merge(sector_poi, on="sector", how="left")

    city_indexes_monthly = _expand_city_indexes(raw_tables["city_indexes"], months)
    panel = panel.merge(city_indexes_monthly, on="month", how="left")

    search_features = _build_search_features(raw_tables["city_search_index"], months)
    panel = panel.merge(search_features, on="month", how="left")

    return panel


def attach_target(panel_df: pd.DataFrame) -> pd.DataFrame:
    if "amount_new_house_transactions" not in panel_df.columns:
        raise KeyError("amount_new_house_transactions column missing from panel")
    panel = panel_df.copy()
    raw_target = panel["amount_new_house_transactions"].fillna(0.0)
    known_mask = panel["amount_new_house_transactions"].notna()
    cutoff_month = panel.loc[known_mask, "month"].max()
    panel["target"] = raw_target
    panel.loc[panel["month"] > cutoff_month, "target"] = np.nan
    panel["target_filled"] = panel["target"].fillna(0.0)
    panel["is_future"] = panel["month"] > cutoff_month
    sector_id_series = panel['sector'].str.extract(r'(\d+)')[0]
    if sector_id_series.isna().any():
        missing = panel.loc[sector_id_series.isna(), 'sector'].unique()
        raise ValueError(f'Unable to parse sector ids for entries: {missing}')
    panel['sector_id'] = sector_id_series.astype(int)
    panel['id'] = panel['month'].dt.strftime('%Y %b') + '_sector ' + panel['sector_id'].astype(str)
    panel = panel.drop(columns=["amount_new_house_transactions"])
    return panel


def save_panel(panel_df: pd.DataFrame, panel_path: Path) -> None:
    panel_path = Path(panel_path)
    if panel_path.exists():
        if panel_path.is_dir():
            for child in panel_path.glob("*"):
                if child.is_dir():
                    import shutil
                    shutil.rmtree(child)
                else:
                    child.unlink()
        else:
            panel_path.unlink()
    panel_df_with_year = panel_df.assign(year=panel_df["month"].dt.year)
    panel_df_with_year.to_parquet(panel_path, index=False, partition_cols=["year"])
