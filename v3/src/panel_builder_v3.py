from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

__all__ = [
    "MONTH_RANGE",
    "TARGET_CUTOFF",
    "FORECAST_MONTHS",
    "load_raw_tables_v3",
    "build_calendar_v3",
    "merge_sources_v3",
    "attach_target_v3",
    "save_panel_v3",
]

MONTH_RANGE = pd.date_range("2019-01-01", "2024-12-01", freq="MS")
TARGET_CUTOFF = pd.Timestamp("2024-07-31")
FORECAST_MONTHS = pd.date_range("2024-08-01", "2024-12-01", freq="MS")


MONTHLY_TABLE_KEYS = (
    "new_house_transactions",
    "new_house_transactions_nearby_sectors",
    "pre_owned_house_transactions",
    "pre_owned_house_transactions_nearby_sectors",
    "land_transactions",
    "land_transactions_nearby_sectors",
)

CITY_INDEX_ALLOWED_COLUMNS = [
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

POI_CORE_COLUMNS = [
    "resident_population",
    "resident_population_dense",
    "population_scale",
    "population_scale_dense",
    "office_population",
    "office_population_dense",
    "surrounding_housing_average_price",
    "surrounding_shop_average_rent",
    "sector_coverage",
]


def _normalize_column_name(name: str) -> str:
    normalized = unicodedata.normalize("NFKC", name).strip()
    normalized = re.sub(r"[^0-9A-Za-z]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    return normalized.strip("_").lower()


def _normalize_text_value(value: object) -> object:
    if pd.isna(value):
        return value
    text = unicodedata.normalize("NFKC", str(value))
    return text.strip()


def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for column in normalized.columns:
        if normalized[column].dtype == object:
            normalized[column] = normalized[column].map(_normalize_text_value)
    return normalized


def _resolve_path(config_path: Path, value: str | Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (config_path.parent / candidate).resolve()


def _slugify_keyword(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKC", text).strip().lower()
    ascii_text = re.sub(r"\s+", "_", ascii_text)
    ascii_text = re.sub(r"[^0-9a-z_]+", "", ascii_text)
    return ascii_text


def load_raw_tables_v3(config_path: Path) -> dict[str, pd.DataFrame]:
    """Load raw CSV tables defined in the v3 data configuration."""

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    train_dir_value = config.get("train_dir")
    if train_dir_value is None:
        raise KeyError("data config must include 'train_dir'")
    train_dir = _resolve_path(config_path, train_dir_value)

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

    dtype_map: Dict[str, Dict[str, str]] = {
        "month_sector": {"month": "string", "sector": "string"},
        "search": {"month": "string", "keyword": "string", "source": "string"},
        "sector": {"sector": "string"},
        "indexes": {"city_indicator_data_year": "Int64"},
    }

    tables: dict[str, pd.DataFrame] = {}
    for key, filename in mapping.items():
        table_path = train_dir / filename
        if not table_path.exists():
            raise FileNotFoundError(f"Missing required data file: {table_path}")

        if key in MONTHLY_TABLE_KEYS:
            dtype = dtype_map["month_sector"]
        elif key == "city_search_index":
            dtype = dtype_map["search"]
        elif key == "sector_poi":
            dtype = dtype_map["sector"]
        elif key == "city_indexes":
            dtype = dtype_map["indexes"]
        else:
            dtype = None

        df = pd.read_csv(table_path, dtype=dtype, encoding="utf-8")
        df.columns = [_normalize_column_name(col) for col in df.columns]
        df = _normalize_dataframe(df)

        if "sector" in df.columns:
            df["sector"] = df["sector"].str.replace(r"\s+", " ", regex=True)

        if key == "city_search_index":
            df["keyword"] = df["keyword"].astype(str).map(_normalize_text_value)
            df["source"] = df["source"].astype(str).map(_normalize_text_value)
            df["keyword_slug"] = df["keyword"].astype(str).map(_slugify_keyword)
        tables[key] = df

    test_path_value = config.get("test_path")
    if test_path_value is None:
        raise KeyError("data config must include 'test_path'")
    test_path = _resolve_path(config_path, test_path_value)
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test file: {test_path}")
    test_df = pd.read_csv(test_path, dtype={"id": "string"}, encoding="utf-8")
    test_df.columns = [_normalize_column_name(col) for col in test_df.columns]
    tables["test"] = _normalize_dataframe(test_df)

    sample_submission_value = config.get("sample_submission")
    if sample_submission_value is None:
        raise KeyError("data config must include 'sample_submission'")
    sample_path = _resolve_path(config_path, sample_submission_value)
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing sample submission file: {sample_path}")
    sample_df = pd.read_csv(sample_path, dtype={"id": "string"}, encoding="utf-8")
    sample_df.columns = [_normalize_column_name(col) for col in sample_df.columns]
    tables["sample_submission"] = _normalize_dataframe(sample_df)

    return tables


def build_calendar_v3(raw_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Construct the full month-sector calendar for the v3 panel."""

    for key in MONTHLY_TABLE_KEYS:
        if key not in raw_tables:
            raise KeyError(f"Missing monthly table '{key}' in raw tables")

    sectors: set[str] = set()
    for key in MONTHLY_TABLE_KEYS:
        table = raw_tables[key]
        if "sector" not in table.columns:
            raise KeyError(f"Table '{key}' is missing the 'sector' column")
        sectors.update(table["sector"].dropna().unique().tolist())

    if "test" in raw_tables and "id" in raw_tables["test"].columns:
        test_ids = raw_tables["test"]["id"].dropna().astype(str)
        extracted = test_ids.str.extract(r"(sector \d+)")[0].dropna().unique()
        sectors.update(extracted)

    sectors = sorted(sectors)
    if not sectors:
        raise ValueError("No sectors discovered while building the calendar")

    calendar = (
        pd.MultiIndex.from_product([MONTH_RANGE, sectors], names=["month", "sector"])
        .to_frame(index=False)
        .sort_values(["month", "sector"], ignore_index=True)
    )

    try:
        sector_id = calendar["sector"].str.extract(r"(\d+)")[0].astype(int)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError("Unable to parse sector identifiers") from exc

    if sector_id.isna().any():
        missing = calendar.loc[sector_id.isna(), "sector"].unique()
        raise ValueError(f"Unable to parse sector identifiers: {missing}")

    calendar["sector_id"] = sector_id.astype(np.int32)
    return calendar


def merge_sources_v3(calendar_df: pd.DataFrame, raw_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge all data sources onto the panel calendar."""

    panel = calendar_df.copy()

    for key in MONTHLY_TABLE_KEYS:
        table = raw_tables[key].copy()
        table["month"] = pd.to_datetime(table["month"], errors="coerce")
        panel = panel.merge(table, on=["month", "sector"], how="left")

    poi = raw_tables["sector_poi"].copy()
    panel = panel.merge(poi, on="sector", how="left")

    city_indexes = _prepare_city_indexes(raw_tables["city_indexes"])
    panel = panel.merge(city_indexes, on="month", how="left")

    search_index = _prepare_search_index(raw_tables["city_search_index"])
    panel = panel.merge(search_index, on="month", how="left")

    panel["month"] = pd.to_datetime(panel["month"], errors="coerce")
    panel = panel.sort_values(["month", "sector"], ignore_index=True)
    return panel


def _prepare_city_indexes(city_df: pd.DataFrame) -> pd.DataFrame:
    df = city_df.copy()
    df["city_indicator_data_year"] = pd.to_numeric(
        df["city_indicator_data_year"], errors="coerce"
    ).astype("Int64")
    df = df.dropna(subset=["city_indicator_data_year"])
    df = df.sort_values("city_indicator_data_year")

    value_cols = [col for col in df.columns if col != "city_indicator_data_year"]
    renamed = {
        col: f"city_{col}" if not col.startswith("city_") else col for col in value_cols
    }
    df = df.rename(columns=renamed)

    selected_cols = [col for col in renamed.values() if col in CITY_INDEX_ALLOWED_COLUMNS]
    missing_cols = sorted(set(CITY_INDEX_ALLOWED_COLUMNS) - set(selected_cols))
    if missing_cols:
        for col in missing_cols:
            df[col] = np.nan
        selected_cols = CITY_INDEX_ALLOWED_COLUMNS

    records = []
    for _, row in df.iterrows():
        year = int(row["city_indicator_data_year"])
        for month in range(1, 13):
            record_month = pd.Timestamp(year=year, month=month, day=1)
            record = {"month": record_month}
            for col in selected_cols:
                record[col] = row.get(col)
                record[f"{col}_was_interpolated"] = 1 if month != 1 else 0
            records.append(record)

    monthly = pd.DataFrame(records)
    monthly = monthly.sort_values("month").drop_duplicates("month", keep="last")
    monthly = monthly.set_index("month").reindex(MONTH_RANGE).ffill().reset_index()
    monthly.rename(columns={"index": "month"}, inplace=True)

    for col in selected_cols:
        flag_col = f"{col}_was_interpolated"
        if flag_col not in monthly.columns:
            monthly[flag_col] = 1
        monthly.loc[monthly["month"].dt.month == 1, flag_col] = 0

    return monthly[["month", *selected_cols, *[f"{c}_was_interpolated" for c in selected_cols]]]


def _prepare_search_index(search_df: pd.DataFrame) -> pd.DataFrame:
    df = search_df.copy()
    if df.empty:
        return pd.DataFrame({"month": MONTH_RANGE})

    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df = df.dropna(subset=["month", "keyword_slug"])
    df = df[df["month"] <= pd.Timestamp("2024-07-31")]
    df["search_volume"] = pd.to_numeric(df.get("search_volume"), errors="coerce")
    df = df.dropna(subset=["search_volume"])

    if df.empty:
        return pd.DataFrame({"month": MONTH_RANGE})

    agg = df.groupby(["month", "keyword_slug"], as_index=False)["search_volume"].sum()
    pivot = (
        agg.pivot(index="month", columns="keyword_slug", values="search_volume")
        .reindex(MONTH_RANGE)
        .reset_index()
    )
    pivot.columns = ["month", *[f"search_kw_{slug}" for slug in pivot.columns[1:]]]
    return pivot


def attach_target_v3(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Attach target-related columns and identifiers to the panel."""

    if "amount_new_house_transactions" not in panel_df.columns:
        raise KeyError("amount_new_house_transactions column missing from panel")

    panel = panel_df.copy()
    panel["month"] = pd.to_datetime(panel["month"], errors="coerce")

    sector_id = panel["sector"].str.extract(r"(\d+)")[0].astype(float)
    if sector_id.isna().any():
        missing = panel.loc[sector_id.isna(), "sector"].unique()
        raise ValueError(f"Unable to parse sector identifiers: {missing}")
    panel["sector_id"] = sector_id.astype(np.int32)

    target_raw = pd.to_numeric(
        panel["amount_new_house_transactions"], errors="coerce"
    )
    target_flag = (panel["month"] <= TARGET_CUTOFF).astype(np.int8)
    panel["target_available_flag"] = target_flag
    panel["target"] = np.where(target_flag == 1, target_raw, np.nan)
    panel["target_filled"] = target_raw.fillna(0.0)
    panel["target_filled_was_missing"] = target_raw.isna().astype(np.int8)
    panel["is_future"] = (panel["month"] > TARGET_CUTOFF).astype(np.int8)

    panel["id"] = panel["month"].dt.strftime("%Y %b") + "_sector " + panel["sector_id"].astype(str)
    panel = panel.drop(columns=["sector"], errors="ignore")
    return panel


def save_panel_v3(panel_df: pd.DataFrame, panel_path: Path) -> None:
    """Persist the panel dataset and accompanying summary report."""

    panel_path = Path(panel_path)
    panel_path.parent.mkdir(parents=True, exist_ok=True)

    panel = panel_df.copy()
    panel["year"] = panel["month"].dt.year.astype(np.int16)

    logger.info("Writing panel dataset to %s", panel_path)
    panel.to_parquet(panel_path, index=False, partition_cols=["year"])

    labeled_mask = panel["target_available_flag"] == 1
    summary = {
        "month_min": panel["month"].min().strftime("%Y-%m-%d"),
        "month_max": panel["month"].max().strftime("%Y-%m-%d"),
        "sector_count": int(panel["sector_id"].nunique()),
        "labeled_rows": int(labeled_mask.sum()),
        "future_rows": int((panel["is_future"] == 1).sum()),
        "missing_target_ratio": float(panel["target"].isna().mean()),
    }

    reports_dir = panel_path.parent.parent / "reports_v3"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "panel_build_summary.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
