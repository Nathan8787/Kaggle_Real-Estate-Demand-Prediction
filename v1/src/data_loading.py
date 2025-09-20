from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import yaml
from unidecode import unidecode


def _to_snake_case(name: str) -> str:
    import re

    name = unicodedata.normalize("NFKC", name).strip()
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name)
    name = re.sub(r"(_)+", "_", name)
    return name.strip("_").lower()


def _normalize_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
    return df


def _normalize_city_search_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["keyword_original"] = df["keyword"]
    df["keyword"] = df["keyword"].apply(lambda x: unicodedata.normalize("NFKC", str(x)))
    df["keyword_slug"] = df["keyword"].apply(lambda x: unidecode(x).replace(" ", "_").lower())
    df["source"] = df["source"].apply(lambda x: unicodedata.normalize("NFKC", str(x)))
    return df


def load_yaml_config(path: Path | str) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def load_raw_tables(config_path: Path | str) -> Dict[str, pd.DataFrame]:
    config = load_yaml_config(config_path)
    config_base = Path(config_path).parent
    train_dir = (config_base / config["train_dir"]).resolve()
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

    tables: Dict[str, pd.DataFrame] = {}
    for key, filename in mapping.items():
        file_path = train_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Expected file not found: {file_path}")
        dtype_map: Dict[str, Any] | None = None
        if key in {"new_house_transactions", "new_house_transactions_nearby_sectors",
                   "pre_owned_house_transactions", "pre_owned_house_transactions_nearby_sectors",
                   "land_transactions", "land_transactions_nearby_sectors"}:
            dtype_map = {"month": "string", "sector": "string"}
        elif key == "sector_poi":
            dtype_map = {"sector": "string"}
        elif key == "city_search_index":
            dtype_map = {"month": "string", "keyword": "string", "source": "string"}
        elif key == "city_indexes":
            dtype_map = {"city_indicator_data_year": "Int64"}

        df = pd.read_csv(file_path, dtype=dtype_map, encoding="utf-8")
        df.columns = [_to_snake_case(col) for col in df.columns]
        df = _normalize_object_columns(df)

        if key == "city_search_index":
            df = _normalize_city_search_index(df)
        if "sector" in df.columns:
            df["sector"] = df["sector"].str.replace("\\s+", " ", regex=True)

        tables[key] = df

    # Load test file for calendar extension
    test_path = (config_base / config.get("test_path", "")).resolve()
    if test_path.exists():
        test_df = pd.read_csv(test_path, dtype={"id": "string"}, encoding="utf-8")
        test_df.columns = [_to_snake_case(col) for col in test_df.columns]
        tables["test"] = test_df

    return tables
