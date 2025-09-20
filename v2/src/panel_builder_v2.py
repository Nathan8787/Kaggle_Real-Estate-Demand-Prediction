from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import yaml
from unidecode import unidecode

__all__ = [
    "load_raw_tables_v2",
    "build_calendar",
    "merge_sources",
    "attach_target_v2",
    "save_panel",
]


MONTH_RANGE_START = pd.Timestamp("2019-01-01")
MONTH_RANGE_END = pd.Timestamp("2025-07-01")
TRAIN_CUTOFF = pd.Timestamp("2024-07-01")


def _to_snake_case(name: str) -> str:
    name = unicodedata.normalize("NFKC", name).strip()
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_").lower()


def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
    return df


def load_raw_tables_v2(config_path: str | Path) -> Dict[str, pd.DataFrame]:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)

    config_dir = config_path.parent
    project_root = config_dir.parent if config_dir.parent != config_dir else config_dir

    def _resolve_path(value: str | Path) -> Path:
        path_obj = Path(value)
        if path_obj.is_absolute():
            return path_obj
        return (project_root / path_obj).resolve(strict=False)

    train_dir = _resolve_path(config["train_dir"])

    mapping = {
        "new_house_transactions": "new_house_transactions.csv",
        "new_house_transactions_nearby_sectors": "new_house_transactions_nearby_sectors.csv",
        "pre_owned_house_transactions": "pre_owned_house_transactions.csv",
        "pre_owned_house_transactions_nearby_sectors": "pre_owned_house_transactions_nearby_sectors.csv",
        "land_transactions": "land_transactions.csv",
        "land_transactions_nearby_sectors": "land_transactions_nearby_sectors.csv",
        "sector_poi": "sector_POI.csv",
        "city_search_index": "city_search_index.csv",
        "city_indexes": "city_indexes.csv",
    }

    dtype_map = {
        "month_str": {"month": "string", "sector": "string"},
        "sector_only": {"sector": "string"},
        "search": {"month": "string", "keyword": "string", "source": "string"},
        "indexes": {"city_indicator_data_year": "Int64"},
    }

    tables: Dict[str, pd.DataFrame] = {}
    for key, filename in mapping.items():
        path = train_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing required training file: {path}")

        if key in {
            "new_house_transactions",
            "new_house_transactions_nearby_sectors",
            "pre_owned_house_transactions",
            "pre_owned_house_transactions_nearby_sectors",
            "land_transactions",
            "land_transactions_nearby_sectors",
        }:
            dtype = dtype_map["month_str"]
        elif key == "sector_poi":
            dtype = dtype_map["sector_only"]
        elif key == "city_search_index":
            dtype = dtype_map["search"]
        elif key == "city_indexes":
            dtype = dtype_map["indexes"]
        else:
            dtype = None

        df = pd.read_csv(path, dtype=dtype, encoding="utf-8")
        df.columns = [_to_snake_case(col) for col in df.columns]
        df = _normalize_dataframe(df)

        if key == "city_search_index":
            df["keyword_original"] = df["keyword"]
            df["keyword"] = df["keyword"].apply(lambda x: unicodedata.normalize("NFKC", str(x)))
            df["keyword_slug"] = df["keyword"].apply(lambda x: unidecode(x).replace(" ", "_").lower())
            df["source"] = df["source"].apply(lambda x: unicodedata.normalize("NFKC", str(x)))
        if "sector" in df.columns:
            df["sector"] = df["sector"].str.replace(r"\s+", " ", regex=True)

        tables[key] = df

    test_path = _resolve_path(config["test_path"])
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test file: {test_path}")
    test_df = pd.read_csv(test_path, dtype={"id": "string"}, encoding="utf-8")
    test_df.columns = [_to_snake_case(col) for col in test_df.columns]
    tables["test"] = test_df

    sample_path = _resolve_path(config["sample_submission"])
    if sample_path.exists():
        sample_df = pd.read_csv(sample_path, encoding="utf-8")
        sample_df.columns = [_to_snake_case(col) for col in sample_df.columns]
        tables["sample_submission"] = sample_df

    return tables


def build_calendar(raw_tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    months = pd.date_range(MONTH_RANGE_START, MONTH_RANGE_END, freq="MS")
    if "new_house_transactions" not in raw_tables:
        raise KeyError("new_house_transactions table is required to build the calendar")
    new_house = raw_tables["new_house_transactions"].copy()
    new_house["month"] = pd.to_datetime(new_house["month"], errors="coerce")
    sectors = {seg for seg in new_house["sector"].dropna().unique()}

    if "test" in raw_tables:
        test_ids = raw_tables["test"].get("id")
        if test_ids is not None:
            extracted = test_ids.str.extract(r"(sector \d+)")
            sectors.update(extracted[0].dropna().unique())

    sectors = sorted(sectors)

    calendar = (
        pd.MultiIndex.from_product([months, sectors], names=["month", "sector"])
        .to_frame(index=False)
        .sort_values(["month", "sector"], ignore_index=True)
    )
    return calendar


def _full_month_index(df: pd.DataFrame, months: Iterable[pd.Timestamp]) -> pd.DataFrame:
    df = df.copy()
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    sectors = df["sector"].dropna().unique()
    full_index = pd.MultiIndex.from_product([sectors, months], names=["sector", "month"])
    numeric_cols = [col for col in df.columns if col not in {"sector", "month"}]
    df = df.set_index(["sector", "month"]).sort_index()
    df = df.reindex(full_index)
    df = df.reset_index()
    ordered_cols = ["month", "sector"] + numeric_cols
    return df[ordered_cols]


def _fill_future_values(table: pd.DataFrame, months: Iterable[pd.Timestamp], cutoff: pd.Timestamp) -> pd.DataFrame:
    df = _full_month_index(table, months)
    df = df.sort_values(["sector", "month"]).reset_index(drop=True)
    numeric_cols = [col for col in df.columns if col not in {"month", "sector"}]

    for col in numeric_cols:
        synthetic_flag = f"{col}_synthetic"
        df[synthetic_flag] = 0

        nan_future_mask = df[col].isna() & (df["month"] > cutoff)

        # Step 1: sector forward fill
        df[col] = df.groupby("sector")[col].ffill()

        # Step 2: sector rolling medians using previous observed values
        roll6 = (
            df.groupby("sector")[col]
            .transform(lambda s: s.shift(1).rolling(window=6, min_periods=4).median())
        )
        roll3 = (
            df.groupby("sector")[col]
            .transform(lambda s: s.shift(1).rolling(window=3, min_periods=2).median())
        )
        df.loc[df[col].isna(), col] = roll6.loc[df[col].isna()]
        df.loc[df[col].isna(), col] = roll3.loc[df[col].isna()]

        # Step 3: city median (per month)
        city_median = df.groupby("month")[col].transform("median")
        df.loc[df[col].isna(), col] = city_median.loc[df[col].isna()]

        # Step 4: global median
        global_median = float(df[col].median(skipna=True)) if not df[col].dropna().empty else 0.0
        df[col] = df[col].fillna(global_median)

        # Step 5: remaining missing -> 0
        df[col] = df[col].fillna(0.0)

        synthetic_rows = nan_future_mask & df[col].notna()
        df.loc[synthetic_rows, synthetic_flag] = 1

    return df


def _extend_monthly_tables(raw_tables: Dict[str, pd.DataFrame], months: Iterable[pd.Timestamp]) -> Dict[str, pd.DataFrame]:
    monthly_keys = [
        "new_house_transactions",
        "new_house_transactions_nearby_sectors",
        "pre_owned_house_transactions",
        "pre_owned_house_transactions_nearby_sectors",
        "land_transactions",
        "land_transactions_nearby_sectors",
    ]

    for key in monthly_keys:
        extended = _fill_future_values(raw_tables[key], months, TRAIN_CUTOFF)
        raw_tables[key] = extended
    return raw_tables


def merge_sources(calendar_df: pd.DataFrame, raw_tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    months = pd.date_range(MONTH_RANGE_START, MONTH_RANGE_END, freq="MS")
    raw_tables = _extend_monthly_tables(raw_tables, months)

    panel = calendar_df.copy()
    merge_sequence: Tuple[Tuple[str, Iterable[str]], ...] = (
        ("new_house_transactions", ["month", "sector"]),
        ("new_house_transactions_nearby_sectors", ["month", "sector"]),
        ("pre_owned_house_transactions", ["month", "sector"]),
        ("pre_owned_house_transactions_nearby_sectors", ["month", "sector"]),
        ("land_transactions", ["month", "sector"]),
        ("land_transactions_nearby_sectors", ["month", "sector"]),
    )

    for key, join_cols in merge_sequence:
        df = raw_tables[key].copy()
        df["month"] = pd.to_datetime(df["month"], errors="coerce")
        panel = panel.merge(df, on=list(join_cols), how="left")

    sector_poi = raw_tables["sector_poi"].copy()
    panel = panel.merge(sector_poi, on="sector", how="left")

    city_monthly = _expand_city_indexes_with_growth(raw_tables["city_indexes"], months)
    panel = panel.merge(city_monthly, on="month", how="left")

    search_features = _build_search_features_v2(raw_tables["city_search_index"], months)
    panel = panel.merge(search_features, on="month", how="left")

    panel["month"] = pd.to_datetime(panel["month"], errors="coerce")
    panel = panel.sort_values(["month", "sector"]).reset_index(drop=True)
    return panel


def _expand_city_indexes_with_growth(city_index_df: pd.DataFrame, months: Iterable[pd.Timestamp]) -> pd.DataFrame:
    df = city_index_df.copy()
    df["city_indicator_data_year"] = df["city_indicator_data_year"].astype(int)
    df = df.sort_values("city_indicator_data_year")
    value_cols = [col for col in df.columns if col != "city_indicator_data_year"]

    year_to_values = {
        int(row["city_indicator_data_year"]): row[value_cols].to_dict()
        for _, row in df.iterrows()
    }
    if not year_to_values:
        raise ValueError("city_indexes table is empty")

    years_sorted = sorted(year_to_values.keys())
    new_rows = []
    for year in range(years_sorted[0], MONTH_RANGE_END.year + 1):
        if year in year_to_values:
            record = {"month": pd.Timestamp(f"{year}-01-01")}
            for col in value_cols:
                record[f"city_{col}"] = year_to_values[year][col]
                record[f"city_{col}_was_interpolated"] = 0
            new_rows.append(record)
            continue

        # Compute growth from last available years
        history_years = [y for y in years_sorted if y < year]
        lookback = history_years[-3:] if len(history_years) >= 3 else history_years
        base_year = history_years[-1]
        record = {"month": pd.Timestamp(f"{year}-01-01")}
        for col in value_cols:
            base_value = year_to_values[base_year][col]
            if not lookback:
                value = base_value
            else:
                first_year = lookback[0]
                first_value = year_to_values[first_year][col]
                if first_value is None or pd.isna(first_value) or first_value <= 0 or base_value <= 0:
                    deltas = [
                        year_to_values[lookback[i + 1]][col] - year_to_values[lookback[i]][col]
                        for i in range(len(lookback) - 1)
                    ]
                    mean_delta = np.nanmean(deltas) if deltas else 0.0
                    value = float(base_value) + mean_delta
                else:
                    periods = len(lookback) - 1 if len(lookback) > 1 else 1
                    cagr = (float(base_value) / float(first_value)) ** (1 / periods)
                    value = float(base_value) * cagr
            record[f"city_{col}"] = value
            record[f"city_{col}_was_interpolated"] = 1
        new_rows.append(record)

    monthly_frame = pd.DataFrame(new_rows)
    monthly_frame = monthly_frame.sort_values("month").reset_index(drop=True)
    monthly_frame = monthly_frame.set_index("month").reindex(months).ffill().reset_index()
    monthly_frame = monthly_frame.rename(columns={"index": "month"})
    return monthly_frame


def _build_search_features_v2(search_df: pd.DataFrame, months: Iterable[pd.Timestamp]) -> pd.DataFrame:
    months_list = list(months)
    months_index = pd.DatetimeIndex(pd.to_datetime(months_list)) if months_list else pd.DatetimeIndex([])
    if search_df.empty:
        return pd.DataFrame({"month": months_index})

    df = search_df.copy()
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df["search_volume"] = pd.to_numeric(df.get("search_volume"), errors="coerce")
    df = df.dropna(subset=["month", "keyword_slug", "search_volume"])
    df["keyword_slug"] = df["keyword_slug"].astype(str).str.strip()
    df = df[df["keyword_slug"] != ""]
    if df.empty:
        return pd.DataFrame({"month": months_index})

    agg = df.groupby(["month", "keyword_slug"], as_index=False)["search_volume"].sum()
    slug_variance = agg.groupby("keyword_slug")["search_volume"].var(ddof=0).fillna(0.0)
    top_slugs = slug_variance.sort_values(ascending=False).head(30).index.tolist()
    if not top_slugs:
        top_slugs = slug_variance.index.tolist()
    if not top_slugs:
        return pd.DataFrame({"month": months_index})

    pivot_raw = (
        agg.pivot(index="month", columns="keyword_slug", values="search_volume")
        .reindex(months_index)
    )
    pivot_selected = pivot_raw.reindex(columns=top_slugs)

    missing_flags = pivot_selected.isna().astype(np.int8)
    smoothed = pivot_selected.ffill().bfill()
    smoothed = smoothed.rolling(window=3, min_periods=1).mean()
    smoothed = smoothed.fillna(0.0)

    feature_cols = [f"search_kw_{slug}" for slug in top_slugs]
    flag_cols = [f"{name}_was_missing" for name in feature_cols]
    smoothed.columns = feature_cols
    missing_flags.columns = flag_cols

    result = pd.concat([smoothed, missing_flags], axis=1).reset_index()
    result.rename(columns={"index": "month"}, inplace=True)
    result["month"] = pd.to_datetime(result["month"], errors="coerce")
    return result

def attach_target_v2(panel_df: pd.DataFrame) -> pd.DataFrame:
    if "amount_new_house_transactions" not in panel_df.columns:
        raise KeyError("amount_new_house_transactions column missing")

    panel = panel_df.copy()
    panel["target"] = panel["amount_new_house_transactions"].fillna(0.0)
    panel["target_filled"] = panel["target"].fillna(0.0)
    panel["is_future"] = panel["month"] > TRAIN_CUTOFF
    panel["target_filled_was_missing"] = (
        panel_df["amount_new_house_transactions"].isna().astype(np.int8)
    )
    panel = panel.drop(columns=["amount_new_house_transactions"], errors="ignore")
    sector_match = panel["sector"].str.extract(r"(\d+)")
    if sector_match.isna().any().bool():
        missing = panel.loc[sector_match[0].isna(), "sector"].unique()
        raise ValueError(f"Unable to parse sector ids for entries: {missing}")
    panel["sector_id"] = sector_match[0].astype(int)
    panel["id"] = panel["month"].dt.strftime("%Y %b") + "_sector " + panel["sector_id"].astype(str)
    return panel


def save_panel(panel_df: pd.DataFrame, panel_path: str | Path) -> None:
    panel_path = Path(panel_path)
    if panel_path.exists():
        if panel_path.is_dir():
            for child in panel_path.glob("**/*"):
                if child.is_file():
                    child.unlink()
            for child in sorted(panel_path.glob("*"), reverse=True):
                if child.is_dir():
                    child.rmdir()
        else:
            panel_path.unlink()
    panel_df = panel_df.copy()
    panel_df["year"] = panel_df["month"].dt.year
    panel_df.to_parquet(panel_path, index=False, partition_cols=["year"])
