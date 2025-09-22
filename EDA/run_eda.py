#!/usr/bin/env python
"""EDA pipeline for the China Real Estate Demand Prediction competition."""

from __future__ import annotations

import json
import logging
import math
import os
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import ruptures as rpt
import seaborn as sns
import shap
import statsmodels.api as sm
import typer
from lightgbm import LGBMRegressor
from matplotlib.font_manager import FontProperties, fontManager
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# 註解：設定 Typer 應用程式以符合 CLI 需求
app = typer.Typer(add_completion=False)


@dataclass
class FileMetadata:
    """Container for file size and SHA256."""

    size: int
    sha256: str


@dataclass
class Config:
    """Pipeline configuration."""

    root_dir: Path
    data_dir: Path
    reports_dir: Path
    spec_version: str = "v2.0"
    spec_date: str = "2024-10-05"
    expected_files: Dict[str, FileMetadata] = field(default_factory=dict)
    font_name: str = "Source Han Sans"
    random_seed: int = 42

    @classmethod
    def create(cls, base_dir: Path) -> "Config":
        """Factory method building the configuration from repository root."""

        data_dir = base_dir / "train"
        reports_dir = base_dir / "EDA" / "reports"
        expected = {
            "train/new_house_transactions.csv": FileMetadata(
                size=371303,
                sha256="5571163fa26af87de928ad6441e476afdf320de03ec80cdc7dc11876775f5114",
            ),
            "train/new_house_transactions_nearby_sectors.csv": FileMetadata(
                size=585630,
                sha256="338939cdeb1f90c629688904da61dd9c7ea6bc87b28e529846b78015af344546",
            ),
            "train/pre_owned_house_transactions.csv": FileMetadata(
                size=236488,
                sha256="1e3e7c5a9f539b6d7adf169b482a1d740a0fbec4302f6fad356050803e565cf3",
            ),
            "train/pre_owned_house_transactions_nearby_sectors.csv": FileMetadata(
                size=316688,
                sha256="ac3de41a6adccc9e1d1ec4c2d5ddfa7e9c21730a1682091e741afbf8339bd634",
            ),
            "train/land_transactions.csv": FileMetadata(
                size=162563,
                sha256="affd9f783917ebb047db4b0ca3fb9f7445f5a01028f59a042ad3b2a7a3697b4f",
            ),
            "train/land_transactions_nearby_sectors.csv": FileMetadata(
                size=169020,
                sha256="fa470ce756d23a8349eaeb2a5ac309b0b8778ff8179c7eb077fab33c950e707e",
            ),
            "train/city_search_index.csv": FileMetadata(
                size=126568,
                sha256="23c3596e8f61ad6cce018efd04d841666f8ae9fb5a40c197934bff1a8f4912cb",
            ),
            "train/city_indexes.csv": FileMetadata(
                size=6119,
                sha256="97bc849b3180517b61b0366693996a25616638b0ad2f64764480a284778d6d05",
            ),
            "train/sector_POI.csv": FileMetadata(
                size=63415,
                sha256="1b2eb7139028797c262ff20968eb347a70bb947995d16a1bfdb9c46d538e53b0",
            ),
            "test.csv": FileMetadata(
                size=22967,
                sha256="762cc524dacfb6bc488c35bafa86c0ff93b7dd65a569245300bf99334e5f8dcb",
            ),
            "sample_submission.csv": FileMetadata(
                size=29879,
                sha256="abf2f25f8f763c5fe06c0f6786e60a828f1d7787d0c40f96f9be99b52964b40a",
            ),
        }
        return cls(
            root_dir=base_dir,
            data_dir=data_dir,
            reports_dir=reports_dir,
            expected_files=expected,
        )


# 註解：自訂例外類型以利錯誤處理
class SpecComplianceError(RuntimeError):
    """Raised when the pipeline cannot satisfy the specification."""


def set_headless_backend(headless: bool) -> None:
    """Configure Matplotlib backend."""

    # 註解：競賽要求支援 headless，因此必要時切換 Agg
    if headless:
        matplotlib.use("Agg")


def configure_logging(log_path: Path) -> None:
    """Configure structured logging."""

    # 註解：建立 logging 設定，確保 CLI 與檔案同時紀錄
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="w"),
        ],
    )


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and LightGBM."""

    # 註解：為確保結果可重現，固定所有隨機來源
    random.seed(seed)
    np.random.seed(seed)
    lgb.register_logger(logging.getLogger(__name__))


def validate_environment(config: Config) -> None:
    """Ensure Python and package versions comply with the spec."""

    # 註解：檢查 Python 版本是否符合 3.10.14，除非允許覆寫
    expected = (3, 10, 14)
    if sys.version_info[:3] != expected:
        allow = os.getenv("EDA_ALLOW_PYTHON_MISMATCH")
        if not allow:
            raise SpecComplianceError(
                f"Python version must be 3.10.14, found {sys.version_info[:3]}"
            )
        logging.warning("Python version mismatch allowed via EDA_ALLOW_PYTHON_MISMATCH")

    # 註解：檢查關鍵套件版本
    required_versions = {
        "pandas": "2.1.4",
        "numpy": "1.26.4",
        "scikit-learn": "1.4.2",
        "statsmodels": "0.14.2",
        "lightgbm": "4.3.0",
        "ruptures": "1.1.9",
        "matplotlib": "3.8.4",
        "seaborn": "0.13.2",
    }
    module_name_map = {"scikit-learn": "sklearn"}
    import importlib

    mismatches = []
    for pkg, expected_version in required_versions.items():
        module_name = module_name_map.get(pkg, pkg.replace("-", "_"))
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", None)
        if version != expected_version:
            mismatches.append((pkg, version, expected_version))
    if mismatches:
        problems = ", ".join(f"{pkg}={got} (expected {exp})" for pkg, got, exp in mismatches)
        raise SpecComplianceError(f"Package version mismatch: {problems}")


def ensure_font_available(font_name: str) -> Optional[FontProperties]:
    """Try to register Source Han Sans; return FontProperties if available."""

    # 註解：若字型已安裝則直接使用，否則記錄警告
    fonts = {Path(f.fname).name for f in fontManager.ttflist}
    if any("SourceHanSans" in fname for fname in fonts) or any(
        font_name.lower() in (Path(f.fname).stem.lower()) for f in fontManager.ttflist
    ):
        logging.info("Source Han Sans detected for plotting.")
        return FontProperties(fname=fontManager.ttflist[0].fname)
    logging.warning(
        "Source Han Sans font not found. Please install it to satisfy the plotting requirement."
    )
    return None


def compute_sha256(path: Path) -> str:
    """Compute SHA256 hash for a file."""

    import hashlib

    # 註解：以區塊方式讀取避免大量記憶體占用
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_data_assets(config: Config) -> pd.DataFrame:
    """Verify file sizes and hashes against the specification."""

    # 註解：逐檔案檢查大小與雜湊值
    rows = []
    for relative_path, meta in config.expected_files.items():
        abs_path = config.root_dir / relative_path
        if not abs_path.exists():
            raise SpecComplianceError(f"Missing required file: {relative_path}")
        actual_size = abs_path.stat().st_size
        actual_hash = compute_sha256(abs_path)
        ok = actual_size == meta.size and actual_hash == meta.sha256
        if not ok:
            raise SpecComplianceError(
                f"File integrity mismatch for {relative_path}: size {actual_size} vs {meta.size}, "
                f"hash {actual_hash} vs {meta.sha256}"
            )
        rows.append(
            {
                "file": relative_path,
                "size": actual_size,
                "sha256": actual_hash,
            }
        )
    df = pd.DataFrame(rows)
    target_path = config.reports_dir / "quality" / "file_integrity.csv"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target_path, index=False)
    return df


def clean_reports_dir(config: Config) -> None:
    """Remove previous report outputs to avoid stale artifacts."""

    # 註解：執行前先清空舊有報告資料夾以免影響結果
    if config.reports_dir.exists():
        shutil.rmtree(config.reports_dir)


def ensure_directories(config: Config) -> None:
    """Create all required output directories."""

    # 註解：依照規格建立輸出目錄
    subdirs = [
        "quality",
        "figures",
        "tables",
        "baseline",
        "search",
        "structural",
        "eda_tables",
        "backtest",
        "environment",
    ]
    for name in subdirs:
        (config.reports_dir / name).mkdir(parents=True, exist_ok=True)


def load_monthly_table(path: Path, dtype_map: Dict[str, str]) -> pd.DataFrame:
    """Load a monthly sector table with consistent parsing."""

    # 註解：統一處理月度面板資料
    df = pd.read_csv(path, dtype=dtype_map, encoding="utf-8-sig")
    df["month"] = pd.to_datetime(df["month"], format="%Y-%b")
    df["sector"] = df["sector"].astype("category")
    df["sector_id"] = df["sector"].str.extract(r"(\d+)").astype("int32")
    return df


def add_zero_flags(df: pd.DataFrame, flag_map: Dict[str, str]) -> pd.DataFrame:
    """Add zero-value flags according to specification."""

    # 註解：針對指定欄位建立零值旗標
    for col, flag in flag_map.items():
        if col in df.columns:
            df[flag] = (df[col].astype(float) == 0).astype("int8")
    return df


def flag_negative_values(df: pd.DataFrame, table_name: str, columns: Iterable[str]) -> pd.DataFrame:
    """Collect rows with negative values for anomaly reporting."""

    # 註解：蒐集負值異常供後續輸出
    records = []
    for col in columns:
        if col not in df.columns:
            continue
        mask = df[col].astype(float) < 0
        if mask.any():
            subset = df.loc[mask, ["month", "sector", col]].copy()
            subset["column"] = col
            subset["issue"] = "negative"
            subset["table"] = table_name
            records.append(subset)
    if records:
        result = pd.concat(records, ignore_index=True)
    else:
        result = pd.DataFrame(columns=["month", "sector", "column", "issue", "table"])
    return result


def save_anomalies(df: pd.DataFrame, config: Config, table_key: str) -> None:
    """Persist anomalies to Parquet."""

    # 註解：依表格名稱輸出異常紀錄
    path = config.reports_dir / "quality" / f"outliers_{table_key}.parquet"
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path)


def summarize_missing_values(dfs: Dict[str, pd.DataFrame], config: Config) -> None:
    """Aggregate missing value statistics across tables."""

    # 註解：計算每表缺值統計
    records = []
    for name, df in dfs.items():
        missing = df.isna().sum()
        for col, count in missing.items():
            if count > 0:
                records.append(
                    {
                        "table": name,
                        "column": col,
                        "missing_count": int(count),
                        "missing_ratio": float(count / len(df)),
                    }
                )
    result = pd.DataFrame(records)
    target = config.reports_dir / "quality" / "missing_values.csv"
    if not result.empty:
        result.to_csv(target, index=False)
    else:
        pd.DataFrame(columns=["table", "column", "missing_count", "missing_ratio"]).to_csv(
            target, index=False
        )


def detect_missing_months(df: pd.DataFrame, table_name: str) -> Dict[str, List[str]]:
    """Identify missing months per sector."""

    # 註解：列出每個 sector 缺少的月份
    expected_months = pd.date_range("2019-01-01", "2024-07-01", freq="MS")
    missing = {}
    for sector, group in df.groupby("sector"):
        present = group["month"].unique()
        missing_months = sorted(set(expected_months) - set(present))
        if missing_months:
            missing[str(sector)] = [m.strftime("%Y-%m-%d") for m in missing_months]
    return {"table": table_name, "missing": missing}


def save_missing_months(reports: List[Dict[str, List[str]]], config: Config) -> None:
    """Persist missing months to JSON."""

    # 註解：寫入所有表的缺月資訊
    data = {report["table"]: report["missing"] for report in reports}
    path = config.reports_dir / "quality" / "missing_months.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_datasets(config: Config) -> Dict[str, pd.DataFrame]:
    """Load all required datasets."""

    # 註解：定義 dtype 映射以降低記憶體
    monthly_numeric = {
        "num_new_house_transactions": "float32",
        "area_new_house_transactions": "float32",
        "price_new_house_transactions": "float32",
        "amount_new_house_transactions": "float32",
        "area_per_unit_new_house_transactions": "float32",
        "total_price_per_unit_new_house_transactions": "float32",
        "num_new_house_available_for_sale": "float32",
        "area_new_house_available_for_sale": "float32",
        "period_new_house_sell_through": "float32",
    }
    new_house = load_monthly_table(config.data_dir / "new_house_transactions.csv", monthly_numeric)
    new_house = add_zero_flags(
        new_house,
        {
            "price_new_house_transactions": "price_was_zero_tx",
            "amount_new_house_transactions": "target_was_zero_tx",
            "total_price_per_unit_new_house_transactions": "avg_price_was_zero",
            "num_new_house_available_for_sale": "inventory_zero_flag",
            "area_new_house_available_for_sale": "inventory_area_zero_flag",
        },
    )

    anomalies_new_house = flag_negative_values(
        new_house,
        "new_house_transactions",
        [
            "num_new_house_transactions",
            "area_new_house_transactions",
            "price_new_house_transactions",
            "amount_new_house_transactions",
            "area_per_unit_new_house_transactions",
            "total_price_per_unit_new_house_transactions",
            "num_new_house_available_for_sale",
            "area_new_house_available_for_sale",
            "period_new_house_sell_through",
        ],
    )
    save_anomalies(anomalies_new_house, config, "new_house_transactions")

    nearby_house = load_monthly_table(
        config.data_dir / "new_house_transactions_nearby_sectors.csv", monthly_numeric
    )

    pre_owned_numeric = {
        "area_pre_owned_house_transactions": "float32",
        "amount_pre_owned_house_transactions": "float32",
        "num_pre_owned_house_transactions": "float32",
        "price_pre_owned_house_transactions": "float32",
    }
    pre_owned = load_monthly_table(
        config.data_dir / "pre_owned_house_transactions.csv", pre_owned_numeric
    )
    pre_owned = add_zero_flags(
        pre_owned,
        {
            "area_pre_owned_house_transactions": "preowned_area_was_zero_tx",
            "amount_pre_owned_house_transactions": "preowned_amount_was_zero_tx",
            "num_pre_owned_house_transactions": "preowned_num_was_zero_tx",
            "price_pre_owned_house_transactions": "preowned_price_was_zero_tx",
        },
    )
    anomalies_pre_owned = flag_negative_values(
        pre_owned,
        "pre_owned_house_transactions",
        [
            "area_pre_owned_house_transactions",
            "amount_pre_owned_house_transactions",
            "num_pre_owned_house_transactions",
            "price_pre_owned_house_transactions",
        ],
    )
    save_anomalies(anomalies_pre_owned, config, "pre_owned_house_transactions")

    nearby_pre_owned = load_monthly_table(
        config.data_dir / "pre_owned_house_transactions_nearby_sectors.csv", pre_owned_numeric
    )

    land_numeric = {
        "num_land_transactions": "float32",
        "construction_area": "float32",
        "planned_building_area": "float32",
        "transaction_amount": "float32",
    }
    land = load_monthly_table(config.data_dir / "land_transactions.csv", land_numeric)
    anomalies_land = flag_negative_values(
        land,
        "land_transactions",
        [
            "num_land_transactions",
            "construction_area",
            "planned_building_area",
            "transaction_amount",
        ],
    )
    save_anomalies(anomalies_land, config, "land_transactions")

    nearby_land = load_monthly_table(
        config.data_dir / "land_transactions_nearby_sectors.csv", land_numeric
    )

    search = pd.read_csv(config.data_dir / "city_search_index.csv", encoding="utf-8-sig")
    search["month"] = pd.to_datetime(search["month"], format="%Y-%b")

    city_indexes = pd.read_csv(config.data_dir / "city_indexes.csv", encoding="utf-8-sig")

    poi = pd.read_csv(config.data_dir / "sector_POI.csv", encoding="utf-8-sig")

    datasets = {
        "new_house_transactions": new_house,
        "new_house_transactions_nearby": nearby_house,
        "pre_owned_house_transactions": pre_owned,
        "pre_owned_house_transactions_nearby": nearby_pre_owned,
        "land_transactions": land,
        "land_transactions_nearby": nearby_land,
        "city_search_index": search,
        "city_indexes": city_indexes,
        "sector_POI": poi,
    }
    summarize_missing_values(datasets, config)
    missing_reports = [
        detect_missing_months(new_house, "new_house_transactions"),
        detect_missing_months(pre_owned, "pre_owned_house_transactions"),
        detect_missing_months(land, "land_transactions"),
    ]
    save_missing_months(missing_reports, config)
    return datasets


def ensure_data_release_log(config: Config) -> None:
    """Ensure docs/data_release_log.md exists and contains spec info."""

    # 註解：若檔案不存在則建立，並提醒填寫 Last updated
    docs_dir = config.root_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    log_path = docs_dir / "data_release_log.md"
    header = f"# Data Release Log\n\n- Spec version {config.spec_version} ({config.spec_date}) processed."
    if log_path.exists():
        text = log_path.read_text(encoding="utf-8")
        if f"Spec version {config.spec_version}" not in text:
            with log_path.open("a", encoding="utf-8") as f:
                f.write("\n" + header + "\n")
    else:
        with log_path.open("w", encoding="utf-8") as f:
            f.write(header + "\n- Kaggle data page Last updated: TODO\n")


def competition_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the two-stage scaled MAPE competition metric."""

    # 註解：依照規格固定輸入為 float64 以確保運算穩定
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must share the same shape")

    # 註解：避免除以零，對分母進行微小裁切
    denominator = np.clip(np.abs(y_true), np.finfo(np.float64).eps, None)
    ape = np.abs(y_pred - y_true) / denominator

    if (ape > 1).mean() > 0.30:
        return 0.0

    mask = ape <= 1
    d = int(mask.sum())
    if d == 0:
        return 0.0

    mape_d = float(ape[mask].mean())
    scale = d / len(y_true)
    score = 1.0 - mape_d / scale
    return float(np.clip(score, 0.0, 1.0))


def create_sector_city_mapping(config: Config, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create sector to city mapping with coverage checks."""

    # 註解：從既有設定載入映射，若不存在則依序編號
    mapping_path = config.root_dir / "v3" / "config" / "sector_city_map_v3.csv"
    if mapping_path.exists():
        mapping = pd.read_csv(mapping_path)
    else:
        sectors = (
            datasets["new_house_transactions"]["sector"].cat.categories.astype(str).tolist()
        )
        mapping = pd.DataFrame({"sector": sectors, "city_id": range(1, len(sectors) + 1)})
    mapping["sector"] = mapping["sector"].astype(str).str.strip()
    train_sectors = (
        datasets["new_house_transactions"]["sector"].astype(str).str.strip().unique()
    )
    test = pd.read_csv(config.root_dir / "test.csv")
    if "sector" not in test.columns:
        test["sector"] = test["id"].astype(str).str.split("_").str[-1]
    test_sectors = test["sector"].astype(str).str.strip().unique()
    mapping_sectors = mapping["sector"].astype(str).str.strip().unique()
    missing_in_map = (set(train_sectors) | set(test_sectors)) - set(mapping_sectors)
    if missing_in_map:
        raise SpecComplianceError(f"Sector-city mapping missing sectors: {sorted(missing_in_map)}")
    target_path = config.reports_dir / "tables" / "sector_city_map.csv"
    mapping.to_csv(target_path, index=False)
    return mapping


def save_table_with_markdown(df: pd.DataFrame, csv_path: Path) -> None:
    """Save DataFrame to CSV and Markdown as required."""

    # 註解：同時輸出 CSV 與 Markdown 版本
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    md_path = csv_path.with_suffix(".md")
    headers = " | ".join(df.columns)
    separator = " | ".join(["---"] * len(df.columns))
    rows = [" | ".join(map(lambda x: str(x) if x == x else "", row)) for row in df.to_numpy()]
    markdown = "\n".join([f"| {headers} |", f"| {separator} |"] + [f"| {row} |" for row in rows])
    md_path.write_text(markdown, encoding="utf-8")


def generate_target_trends(
    new_house: pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    """Generate target trend plots and statistics."""

    # 註解：彙總總體金額以繪製趨勢與衍生指標
    agg = new_house.groupby("month", as_index=False)["amount_new_house_transactions"].sum()
    agg = agg.sort_values("month").reset_index(drop=True)
    agg["mom"] = agg["amount_new_house_transactions"].pct_change()
    agg["yoy"] = agg["amount_new_house_transactions"].pct_change(12)
    agg["rolling_12"] = agg["amount_new_house_transactions"].rolling(12).mean()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(agg["month"], agg["amount_new_house_transactions"], label="Total Amount")
    ax.plot(agg["month"], agg["rolling_12"], label="Rolling 12M", linestyle="--")
    ax.set_title("New House Transaction Amount Trend", fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Amount (10k RMB)")
    ax.legend()
    fig.autofmt_xdate()
    fig_path = config.reports_dir / "figures" / "target_amount_trend.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 註解：建立 sector 小倍數圖
    sectors = new_house["sector"].cat.categories.astype(str).tolist()
    per_page = 16
    total_pages = math.ceil(len(sectors) / per_page)
    for page in range(total_pages):
        subset_sectors = sectors[page * per_page : (page + 1) * per_page]
        fig, axes = plt.subplots(4, 4, figsize=(16, 12), sharex=True)
        for ax, sector in zip(axes.flatten(), subset_sectors):
            data = new_house[new_house["sector"].astype(str) == sector]
            ax.plot(data["month"], data["amount_new_house_transactions"], color="#1f77b4")
            ax.set_title(sector)
            ax.set_xlabel("Month")
            ax.set_ylabel("Amount")
        for ax in axes.flatten()[len(subset_sectors) :]:
            ax.axis("off")
        fig.suptitle(f"New House Amount by Sector (Page {page + 1})", fontsize=16)
        fig.autofmt_xdate()
        fig.savefig(
            config.reports_dir
            / "figures"
            / f"target_sector_trends_page_{page + 1}.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

    # 註解：計算 YoY 前後五名
    new_house = new_house.sort_values("month")
    new_house["yoy"] = (
        new_house.groupby("sector")["amount_new_house_transactions"].pct_change(12)
    )
    records = []
    for month, group in new_house.groupby("month"):
        valid = group.dropna(subset=["yoy"])
        if valid.empty:
            continue
        top = valid.nlargest(5, "yoy")
        bottom = valid.nsmallest(5, "yoy")
        for rank, row in enumerate(top.itertuples(), start=1):
            records.append(
                {
                    "month": month.strftime("%Y-%m"),
                    "rank": rank,
                    "sector": row.sector,
                    "yoy_change": row.yoy,
                    "type": "Top",
                }
            )
        for rank, row in enumerate(bottom.itertuples(), start=1):
            records.append(
                {
                    "month": month.strftime("%Y-%m"),
                    "rank": rank,
                    "sector": row.sector,
                    "yoy_change": row.yoy,
                    "type": "Bottom",
                }
            )
    extremes = pd.DataFrame(records)
    save_table_with_markdown(
        extremes,
        config.reports_dir / "eda_tables" / "target_yoy_extremes.csv",
    )

    # 註解：建立 lag1 模擬以評估歷史 APE 與競賽分數
    pivot = new_house.pivot_table(
        index="month", columns="sector", values="amount_new_house_transactions"
    )
    predictions = pivot.shift(1)
    ape = np.abs(predictions - pivot) / pivot
    ape = ape.replace([np.inf, -np.inf], np.nan)
    avg_ape = ape.mean(axis=1)
    scores = []
    for month in pivot.index:
        paired = pd.concat(
            [
                pivot.loc[month],
                predictions.loc[month],
            ],
            axis=1,
            keys=["true", "pred"],
        ).dropna()
        if not paired.empty:
            score = competition_score(paired["true"].values, paired["pred"].values)
        else:
            score = np.nan
        scores.append(score)
    history = pd.DataFrame(
        {
            "month": pivot.index.strftime("%Y-%m"),
            "avg_ape": avg_ape.values,
            "competition_score": scores,
        }
    )
    save_table_with_markdown(
        history,
        config.reports_dir / "eda_tables" / "target_backtest_metrics.csv",
    )
    return agg


def generate_preowned_analysis(
    new_house: pd.DataFrame,
    pre_owned: pd.DataFrame,
    config: Config,
) -> None:
    """Analyze interactions between new and pre-owned markets."""

    # 註解：彙總各市場金額
    new_agg = (
        new_house.groupby("month", as_index=False)["amount_new_house_transactions"].sum()
    )
    pre_agg = (
        pre_owned.groupby("month", as_index=False)["amount_pre_owned_house_transactions"].sum()
    )
    merged = pd.merge(new_agg, pre_agg, on="month", how="inner")

    # 註解：計算不同比滯後的相關
    lags = range(0, 7)
    corr_matrix = pd.DataFrame(index=lags, columns=["correlation"])
    for lag in lags:
        shifted = pre_agg.set_index("month")["amount_pre_owned_house_transactions"].shift(lag)
        aligned = pd.concat(
            [
                new_agg.set_index("month")["amount_new_house_transactions"],
                shifted,
            ],
            axis=1,
            join="inner",
        ).dropna()
        if not aligned.empty:
            corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
        else:
            corr = np.nan
        corr_matrix.loc[lag, "correlation"] = corr
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        corr_matrix.T.astype(float),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar_kws={"label": "Pearson R"},
        ax=ax,
    )
    ax.set_xlabel("Lag (months)")
    ax.set_ylabel("Metric")
    ax.set_title("New vs Pre-owned Correlation by Lag")
    fig.savefig(config.reports_dir / "figures" / "preowned_correlation_heatmap.png", dpi=300)
    plt.close(fig)

    merged["volume_ratio"] = (
        merged["amount_new_house_transactions"] / merged["amount_pre_owned_house_transactions"]
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(merged["month"], merged["volume_ratio"], label="Volume Ratio")
    ax.set_title("New to Pre-owned Amount Ratio")
    ax.set_xlabel("Month")
    ax.set_ylabel("Ratio")
    fig.autofmt_xdate()
    fig.savefig(config.reports_dir / "figures" / "new_preowned_ratio_trend.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        np.log1p(merged["amount_pre_owned_house_transactions"]),
        np.log1p(merged["amount_new_house_transactions"]),
        alpha=0.7,
    )
    ax.set_xlabel("log1p Pre-owned Amount")
    ax.set_ylabel("log1p New Amount")
    ax.set_title("Scatter of New vs Pre-owned Amounts")
    fig.savefig(config.reports_dir / "figures" / "new_vs_preowned_scatter.png", dpi=300)
    plt.close(fig)


def generate_land_analysis(
    new_house: pd.DataFrame,
    land: pd.DataFrame,
    config: Config,
) -> None:
    """Analyze land supply impact on new house market."""

    # 註解：聚合土地與新房金額
    new_agg = new_house.groupby("month", as_index=False)["amount_new_house_transactions"].sum()
    land_agg = land.groupby("month", as_index=False)["transaction_amount"].sum()
    merged = pd.merge(new_agg, land_agg, on="month", how="inner")

    # 註解：計算交叉相關
    lags = range(0, 7)
    correlations = []
    for lag in lags:
        shifted = land_agg.set_index("month")["transaction_amount"].shift(lag)
        aligned = pd.concat(
            [
                new_agg.set_index("month")["amount_new_house_transactions"],
                shifted,
            ],
            axis=1,
            join="inner",
        ).dropna()
        corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1]) if not aligned.empty else np.nan
        correlations.append({"lag": lag, "correlation": corr})
    corr_df = pd.DataFrame(correlations)
    save_table_with_markdown(
        corr_df,
        config.reports_dir / "eda_tables" / "land_newhouse_cross_correlation.csv",
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=corr_df, x="lag", y="correlation", ax=ax, color="#1f77b4")
    ax.set_title("Land Transaction Amount vs New House Amount")
    ax.set_xlabel("Lag (months)")
    ax.set_ylabel("Pearson R")
    fig.savefig(config.reports_dir / "figures" / "land_correlation.png", dpi=300)
    plt.close(fig)

    # 註解：計算土地供給為零的連續月份
    land_zero = land[land["transaction_amount"] == 0]
    streaks = []
    for sector, group in land_zero.groupby("sector"):
        months = sorted(group["month"].unique())
        if not months:
            continue
        streak = 1
        prev = months[0]
        for current in months[1:]:
            if current == prev + pd.offsets.MonthBegin(1):
                streak += 1
            else:
                streaks.append({"sector": sector, "streak_length": streak, "end_month": prev})
                streak = 1
            prev = current
        streaks.append({"sector": sector, "streak_length": streak, "end_month": prev})
    streak_df = pd.DataFrame(streaks)
    save_table_with_markdown(
        streak_df,
        config.reports_dir / "eda_tables" / "land_zero_streaks.csv",
    )

    # 註解：使用滯後回歸評估土地對新房金額影響
    land_features = []
    for lag in range(1, 7):
        land_agg[f"land_amount_lag{lag}"] = land_agg["transaction_amount"].shift(lag)
    merged_features = pd.merge(new_agg, land_agg.drop(columns="transaction_amount"), on="month")
    merged_features = merged_features.dropna()
    y = merged_features["amount_new_house_transactions"]
    X = merged_features[[f"land_amount_lag{lag}" for lag in range(1, 7)]]
    model = sm.OLS(y, sm.add_constant(X)).fit()
    params = pd.DataFrame(
        {
            "parameter": model.params.index,
            "estimate": model.params.values,
            "p_value": model.pvalues.values,
        }
    )
    save_table_with_markdown(
        params,
        config.reports_dir / "eda_tables" / "land_lag_regression_coefficients.csv",
    )


def prepare_keyword_slug_mapping() -> Dict[str, str]:
    """Return mapping from Chinese keyword to slug."""

    return {
        "买房": "maifang",
        "二手房市场": "ershouchang",
        "公积金": "gongjijin",
        "限购政策": "xiangou",
        "房贷利率": "fangdaililv",
        "住房补贴": "zhufangbutie",
        "学区房": "xuequfang",
        "装修攻略": "zhuangxiugonglue",
        "房价走势": "fangjia_zoushi",
        "土地拍卖": "tudi_paimai",
        "新房开盘": "xinfang_kaipan",
        "二手房挂牌": "ershoufang_guapai",
        "购房资格": "goufang_zige",
        "预售证": "yushouzheng",
        "楼市政策": "loushi_zhengce",
        "房产税": "fangchan_shui",
        "租赁市场": "zulin_shichang",
        "长租公寓": "changzu_gongyu",
        "商业地产": "shangye_dichan",
        "办公楼租金": "bangonglou_zujin",
        "学区划片": "xuequ_huapian",
        "租房补贴": "zufang_butie",
        "二手房贷款": "ershoufang_daikuan",
        "房贷政策": "fangdai_zhengce",
        "房屋交易税费": "fangwu_shui",
        "租售同权": "zushao_tongquan",
        "二手房过户": "ershoufang_guohu",
        "学区调整": "xuequ_tiaozheng",
        "房屋限售": "fangwu_xianshou",
        "商住两用": "shangzhu_liangyong",
        "共有产权": "gongyou_chanquan",
        "公寓投资": "gongyu_touzi",
        "房地产调控": "fangdichan_tiaokong",
        "新房认购": "xinfang_rengou",
        "购房落户": "goufang_luohu",
    }


def _derive_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Generate lag, rolling, and change based features for a DataFrame."""

    # 註解：為每個欄位建立滯後與統計特徵
    result = frame.copy()
    for col in frame.columns:
        series = frame[col]
        result[f"{col}_lag_1"] = series.shift(1)
        result[f"{col}_lag_3"] = series.shift(3)
        result[f"{col}_lag_6"] = series.shift(6)
        result[f"{col}_rolling_mean_3"] = series.rolling(3).mean()
        result[f"{col}_rolling_std_3"] = series.rolling(3).std()
        result[f"{col}_pct_change_1"] = series.pct_change()
        result[f"{col}_zscore_3"] = (series - series.rolling(3).mean()) / series.rolling(3).std()
    return result


def prepare_search_analysis(
    search_df: pd.DataFrame,
    new_house: pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    """Process search data, generate features, and create visualisations."""

    keyword_map = prepare_keyword_slug_mapping()
    df = search_df.copy()
    df["keyword_slug"] = df["keyword"].map(keyword_map)
    df["keyword_unknown_flag"] = df["keyword_slug"].isna().astype("int8")
    unknown_mask = df["keyword_slug"].isna()
    if unknown_mask.any():
        sanitized = (
            df.loc[unknown_mask, "keyword"]
            .str.lower()
            .str.replace(r"\s+", "_", regex=True)
        )
        df.loc[unknown_mask, "keyword_slug"] = sanitized
        unknown_records = df.loc[unknown_mask, ["month", "keyword", "source"]]
        quality_path = config.reports_dir / "quality" / "search_unknown_keywords.csv"
        unknown_records.to_csv(quality_path, index=False)

    source_map = {"PC端": "pc", "移动端": "mobile"}
    df["source_slug"] = df["source"].map(source_map)
    if df["source_slug"].isna().any():
        raise SpecComplianceError("Unexpected search source encountered")

    df["column_name"] = "search_" + df["keyword_slug"] + "_" + df["source_slug"]
    pivot = (
        df.pivot_table(
            index="month",
            columns="column_name",
            values="search_volume",
            aggfunc="sum",
        )
        .sort_index()
        .fillna(0.0)
    )

    df["total_per_source"] = df.groupby(["month", "source_slug"])["search_volume"].transform("sum")
    df["share"] = np.where(df["total_per_source"] == 0, 0.0, df["search_volume"] / df["total_per_source"])
    share_mean = df.groupby("keyword_slug")["share"].mean().sort_values(ascending=False)

    series_by_slug = {}
    for slug in share_mean.index:
        cols = [c for c in pivot.columns if c.startswith(f"search_{slug}_")]
        if cols:
            series_by_slug[slug] = pivot[cols].sum(axis=1)

    selected_slugs: List[str] = []
    for slug in share_mean.index:
        if slug not in series_by_slug:
            continue
        candidate = series_by_slug[slug]
        correlated = False
        for kept in selected_slugs:
            corr = candidate.corr(series_by_slug[kept])
            if corr is not None and not np.isnan(corr) and abs(corr) > 0.98:
                correlated = True
                break
        if not correlated:
            selected_slugs.append(slug)
        if len(selected_slugs) == 12:
            break

    long_tail_slugs = [slug for slug in series_by_slug if slug not in selected_slugs]

    top_keyword_table = pd.DataFrame(
        {
            "keyword_slug": selected_slugs,
            "average_share": [float(share_mean[slug]) for slug in selected_slugs],
        }
    )
    save_table_with_markdown(
        top_keyword_table,
        config.reports_dir / "eda_tables" / "search_top_keywords.csv",
    )

    long_tail_features = pd.DataFrame(index=pivot.index)
    for source_slug in ["pc", "mobile"]:
        cols = [f"search_{slug}_{source_slug}" for slug in long_tail_slugs if f"search_{slug}_{source_slug}" in pivot.columns]
        if cols:
            long_tail_features[f"search_long_tail_{source_slug}"] = pivot[cols].sum(axis=1)
        else:
            long_tail_features[f"search_long_tail_{source_slug}"] = 0.0
    long_tail_features["search_long_tail_total"] = long_tail_features.sum(axis=1)

    long_tail_matrix = (
        df[df["keyword_slug"].isin(long_tail_slugs)]
        .groupby(["month", "keyword_slug"])["search_volume"]
        .sum()
        .unstack(fill_value=0.0)
    )
    if long_tail_matrix.empty:
        long_tail_matrix = pd.DataFrame(0.0, index=pivot.index, columns=["placeholder"])
    long_tail_share = long_tail_matrix.div(long_tail_matrix.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    nmf_model = NMF(n_components=5, init="nndsvda", random_state=config.random_seed, max_iter=500)
    topic_values = nmf_model.fit_transform(long_tail_share)
    topic_df = pd.DataFrame(
        topic_values,
        index=long_tail_share.index,
        columns=[f"search_topic_{i+1}" for i in range(topic_values.shape[1])],
    )

    scaler = StandardScaler()
    scaled = scaler.fit_transform(pivot)
    pca_model = PCA(n_components=5, whiten=False, random_state=config.random_seed)
    pca_components = pca_model.fit_transform(scaled)
    pca_df = pd.DataFrame(
        pca_components,
        index=pivot.index,
        columns=[f"search_pca_{i+1}" for i in range(pca_components.shape[1])],
    )
    explained_path = config.reports_dir / "search" / "pca_explained_variance.json"
    explained_path.write_text(
        json.dumps({"explained_variance_ratio": pca_model.explained_variance_ratio_.tolist()}, indent=2),
        encoding="utf-8",
    )

    features = pd.DataFrame(index=pivot.index)
    for slug in selected_slugs:
        for source_slug in ["pc", "mobile"]:
            col = f"search_{slug}_{source_slug}"
            if col in pivot.columns:
                features[col] = pivot[col]
        cols = [c for c in pivot.columns if c.startswith(f"search_{slug}_")]
        if cols:
            features[f"search_{slug}_total"] = pivot[cols].sum(axis=1)

    features["search_total"] = pivot.sum(axis=1)
    features = pd.concat([features, long_tail_features, topic_df, pca_df], axis=1)
    enriched = _derive_time_features(features)
    enriched = enriched.sort_index()

    output_path = config.reports_dir / "search" / "search_features.parquet"
    table = pa.Table.from_pandas(enriched.reset_index().rename(columns={"index": "month"}))
    pq.write_table(table, output_path)

    # 註解：生成關鍵字與主題趨勢圖
    top_plot_df = pd.DataFrame({slug: series_by_slug[slug] for slug in selected_slugs})
    fig, axes = plt.subplots(3, 4, figsize=(20, 12), sharex=True)
    for ax, slug in zip(axes.flatten(), selected_slugs):
        ax.plot(top_plot_df.index, top_plot_df[slug])
        ax.set_title(slug)
        ax.set_xlabel("Month")
        ax.set_ylabel("Search Volume")
    for ax in axes.flatten()[len(selected_slugs) :]:
        ax.axis("off")
    fig.suptitle("Top Search Keywords", fontsize=18)
    fig.autofmt_xdate()
    fig.savefig(config.reports_dir / "figures" / "search_top_keywords.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    for col in topic_df.columns:
        ax.plot(topic_df.index, topic_df[col], label=col)
    ax.set_title("Search Topic Components")
    ax.set_xlabel("Month")
    ax.set_ylabel("Topic Intensity")
    ax.legend()
    fig.autofmt_xdate()
    fig.savefig(config.reports_dir / "figures" / "search_topic_trends.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    total_search = features["search_total"]
    agg_new = new_house.groupby("month")["amount_new_house_transactions"].sum()
    aligned = pd.concat([total_search, agg_new], axis=1).dropna()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(aligned["search_total"], aligned["amount_new_house_transactions"], alpha=0.7)
    ax.set_xlabel("Search Total Volume")
    ax.set_ylabel("New House Amount")
    ax.set_title("Search vs New House Amount")
    fig.savefig(config.reports_dir / "figures" / "search_vs_target_scatter.png", dpi=300)
    plt.close(fig)

    return enriched


def process_macro_indicators(
    city_indexes: pd.DataFrame,
    new_house: pd.DataFrame,
    config: Config,
) -> None:
    """Process macro indicators and visualise trends."""

    df = city_indexes.copy()
    df = df.rename(columns={"city_indicator_data_year": "year"})
    df["year"] = df["year"].astype(int)
    df = df.sort_values("year").drop_duplicates("year", keep="first").set_index("year")
    target_years = range(2017, 2025)
    df = df.reindex(target_years)
    original = df.copy()
    df = df.interpolate(method="linear", axis=0, limit_direction="both").ffill().bfill()
    interpolated_mask = original.isna() & ~df.isna()
    summary = interpolated_mask.sum().reset_index()
    summary.columns = ["column", "interpolated_count"]
    save_table_with_markdown(
        summary,
        config.reports_dir / "eda_tables" / "macro_interpolation_summary.csv",
    )

    flags = interpolated_mask.astype(int)
    flags["year"] = flags.index
    save_table_with_markdown(
        flags.reset_index(drop=True),
        config.reports_dir / "eda_tables" / "macro_interpolated_flags.csv",
    )

    selected_metrics = [
        "gdp_100m",
        "real_estate_development_investment_completed_10k",
        "per_capita_disposable_income_absolute_yuan",
        "annual_average_wage_urban_non_private_employees_yuan",
    ]
    available_metrics = [m for m in selected_metrics if m in df.columns]
    if available_metrics:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        for ax, metric in zip(axes.flatten(), available_metrics):
            ax.plot(df.index, df[metric])
            ax.set_title(metric)
            ax.set_xlabel("Year")
            ax.set_ylabel("Value")
        for ax in axes.flatten()[len(available_metrics) :]:
            ax.axis("off")
        fig.suptitle("Macro Indicator Trends", fontsize=18)
        fig.savefig(
            config.reports_dir / "figures" / "macro_indicator_trends.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    new_house_yearly = (
        new_house.assign(year=new_house["month"].dt.year)
        .groupby("year")["amount_new_house_transactions"]
        .mean()
    )
    scatter_df = df.join(new_house_yearly.rename("avg_amount"), how="inner")
    if not scatter_df.empty and available_metrics:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        for ax, metric in zip(axes.flatten(), available_metrics):
            ax.scatter(scatter_df[metric], scatter_df["avg_amount"], alpha=0.7)
            ax.set_xlabel(metric)
            ax.set_ylabel("Average Amount")
            ax.set_title(f"{metric} vs Avg Amount")
        for ax in axes.flatten()[len(available_metrics) :]:
            ax.axis("off")
        fig.savefig(
            config.reports_dir / "figures" / "macro_vs_amount_scatter.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def analyze_poi_features(poi_df: pd.DataFrame, config: Config) -> None:
    """Perform PCA, clustering, and distribution analysis on POI data."""

    df = poi_df.copy()
    df["sector"] = df["sector"].astype(str)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numeric_cols] = df[numeric_cols].fillna(0.0)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[numeric_cols])

    pca_model = PCA(n_components=2, random_state=42)
    components = pca_model.fit_transform(scaled)
    comp_df = pd.DataFrame(
        components,
        columns=["poi_pca_1", "poi_pca_2"],
    )
    comp_df["sector"] = df["sector"].values
    comp_df["sector_id"] = comp_df["sector"].str.extract(r"(\d+)").astype(int)
    save_table_with_markdown(
        comp_df,
        config.reports_dir / "eda_tables" / "poi_pca_components.csv",
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=comp_df, x="poi_pca_1", y="poi_pca_2", ax=ax)
    ax.set_title("POI PCA (2 Components)")
    fig.savefig(config.reports_dir / "figures" / "poi_pca_scatter.png", dpi=300)
    plt.close(fig)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled)
    cluster_df = pd.DataFrame({"sector": df["sector"], "cluster": clusters})
    save_table_with_markdown(
        cluster_df,
        config.reports_dir / "eda_tables" / "poi_kmeans_clusters.csv",
    )

    selected_metrics = numeric_cols[:8]
    if selected_metrics:
        melted = df[["sector"] + selected_metrics].melt(id_vars="sector", var_name="metric", value_name="value")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=melted, x="metric", y="value", ax=ax)
        ax.set_title("POI Feature Distributions")
        ax.tick_params(axis="x", rotation=45)
        fig.savefig(config.reports_dir / "figures" / "poi_feature_boxplots.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def detect_structural_breaks(new_house: pd.DataFrame, config: Config) -> None:
    """Detect structural breaks using binary segmentation and CUSUM tests."""

    overall_series = (
        new_house.groupby("month")["amount_new_house_transactions"].sum().sort_index()
    )
    algo = rpt.Binseg(model="l2", min_size=6, jump=1).fit(overall_series.values)
    penalty = 3 * np.log(len(overall_series))
    breakpoints = algo.predict(pen=penalty)
    break_months = [overall_series.index[i - 1].strftime("%Y-%m") for i in breakpoints[:-1]]

    time = np.arange(len(overall_series))
    X = sm.add_constant(time)
    ols_model = sm.OLS(overall_series.values, X).fit()
    cusum_stat, cusum_pvalue, _ = sm.stats.diagnostic.breaks_cusumolsresid(ols_model.resid, ddof=1)

    sector_totals = (
        new_house.groupby("sector")["amount_new_house_transactions"].sum().sort_values(ascending=False)
    )
    top_sectors = sector_totals.head(5).index.tolist()
    sector_results = []
    for sector in top_sectors:
        sector_series = (
            new_house[new_house["sector"] == sector]
            .sort_values("month")
            .set_index("month")["amount_new_house_transactions"]
        )
        if len(sector_series) < 12:
            continue
        sector_algo = rpt.Binseg(model="l2", min_size=6, jump=1).fit(sector_series.values)
        sector_breaks = sector_algo.predict(pen=penalty)
        sector_break_months = [
            sector_series.index[i - 1].strftime("%Y-%m") for i in sector_breaks[:-1]
        ]
        time_s = np.arange(len(sector_series))
        X_s = sm.add_constant(time_s)
        ols_s = sm.OLS(sector_series.values, X_s).fit()
        cusum_s, cusum_p_s, _ = sm.stats.diagnostic.breaks_cusumolsresid(ols_s.resid, ddof=1)
        sector_results.append(
            {
                "sector": sector,
                "breaks": sector_break_months,
                "cusum_stat": float(cusum_s),
                "cusum_pvalue": float(cusum_p_s),
            }
        )

    output = {
        "overall": {
            "breaks": break_months,
            "cusum_stat": float(cusum_stat),
            "cusum_pvalue": float(cusum_pvalue),
        },
        "sectors": sector_results,
    }
    structural_path = config.reports_dir / "structural" / "structural_breaks.json"
    structural_path.write_text(json.dumps(output, indent=2), encoding="utf-8")


def analyze_target_decomposition(new_house: pd.DataFrame, config: Config) -> None:
    """Decompose target into quantity and price contributions."""

    df = new_house[
        [
            "amount_new_house_transactions",
            "num_new_house_transactions",
            "price_new_house_transactions",
        ]
    ].dropna()
    X = df[["num_new_house_transactions", "price_new_house_transactions"]]
    y = df["amount_new_house_transactions"]
    model = LinearRegression()
    model.fit(X, y)

    coeff_df = pd.DataFrame(
        {
            "feature": ["num_new_house_transactions", "price_new_house_transactions", "intercept"],
            "coefficient": [
                float(model.coef_[0]),
                float(model.coef_[1]),
                float(model.intercept_),
            ],
        }
    )
    save_table_with_markdown(
        coeff_df,
        config.reports_dir / "eda_tables" / "target_decomposition_coefficients.csv",
    )

    explainer = shap.LinearExplainer(model, X, feature_dependence="independent")
    shap_values = explainer.shap_values(X)
    mean_abs = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame(
        {
            "feature": X.columns,
            "mean_abs_shap": mean_abs,
        }
    )
    save_table_with_markdown(
        shap_df,
        config.reports_dir / "eda_tables" / "target_decomposition_shap.csv",
    )

    shap.summary_plot(shap_values, X, show=False)
    plt.title("SHAP Summary for Target Decomposition")
    plt.tight_layout()
    plt.savefig(config.reports_dir / "figures" / "target_decomposition_shap.png", dpi=300)
    plt.close()


def _evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    """Utility to compute competition metrics and ratios."""

    mask = y_true != 0
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return 0.0, float("nan"), float("nan")
    ape = np.abs(y_pred - y_true) / np.abs(y_true)
    ape_gt1_ratio = float((ape > 1).mean())
    d_ratio = float((ape <= 1).mean())
    score = competition_score(y_true, y_pred)
    return score, d_ratio, ape_gt1_ratio


def run_baseline_models(
    new_house: pd.DataFrame,
    land: pd.DataFrame,
    search_features: pd.DataFrame,
    config: Config,
) -> None:
    """Build baseline models, perform backtests, and create reports."""

    panel = new_house[
        [
            "month",
            "sector",
            "sector_id",
            "amount_new_house_transactions",
        ]
    ].copy()
    panel = panel.sort_values(["sector_id", "month"])
    for lag in [1, 3, 6]:
        panel[f"lag_{lag}"] = panel.groupby("sector_id")["amount_new_house_transactions"].shift(lag)
    panel["rolling_mean_3"] = (
        panel.groupby("sector_id")["amount_new_house_transactions"].rolling(3).mean().reset_index(level=0, drop=True)
    )
    panel["rolling_mean_6"] = (
        panel.groupby("sector_id")["amount_new_house_transactions"].rolling(6).mean().reset_index(level=0, drop=True)
    )
    panel["rolling_std_3"] = (
        panel.groupby("sector_id")["amount_new_house_transactions"].rolling(3).std().reset_index(level=0, drop=True)
    )

    search_total = search_features[["search_total"]].copy()
    search_total.index = pd.to_datetime(search_total.index)
    search_total["search_total_lag1"] = search_total["search_total"].shift(1)
    panel = panel.merge(
        search_total[["search_total_lag1"]],
        left_on="month",
        right_index=True,
        how="left",
    )

    land_panel = land[["month", "sector", "transaction_amount"]].copy()
    land_panel = land_panel.sort_values(["sector", "month"])
    land_panel["land_amount_lag3"] = land_panel.groupby("sector")["transaction_amount"].shift(3)
    panel = panel.merge(
        land_panel[["month", "sector", "land_amount_lag3"]],
        on=["month", "sector"],
        how="left",
    )

    feature_cols = [
        "lag_1",
        "lag_3",
        "lag_6",
        "rolling_mean_3",
        "rolling_mean_6",
        "search_total_lag1",
        "land_amount_lag3",
        "sector_id",
    ]

    fold_definitions = [
        ("F1", "2019-01", "2023-12", "2024-01", "2024-01"),
        ("F2", "2019-01", "2024-01", "2024-02", "2024-02"),
        ("F3", "2019-01", "2024-02", "2024-03", "2024-03"),
        ("F4", "2019-01", "2024-03", "2024-04", "2024-04"),
        ("F5", "2019-01", "2024-04", "2024-05", "2024-05"),
        ("F6", "2019-01", "2024-05", "2024-06", "2024-06"),
        ("Holdout", "2019-01", "2024-06", "2024-07", "2024-07"),
    ]

    records = []
    for fold_name, train_start, train_end, val_start, val_end in fold_definitions:
        train_start_dt = pd.to_datetime(train_start)
        train_end_dt = pd.to_datetime(train_end)
        val_start_dt = pd.to_datetime(val_start)
        val_end_dt = pd.to_datetime(val_end)
        train_mask = (panel["month"] >= train_start_dt) & (panel["month"] <= train_end_dt)
        val_mask = (panel["month"] >= val_start_dt) & (panel["month"] <= val_end_dt)
        train_data = panel.loc[train_mask].copy()
        val_data = panel.loc[val_mask].copy()
        purge_months = pd.date_range(val_start_dt - pd.offsets.MonthBegin(1), val_end_dt - pd.offsets.MonthBegin(1), freq="MS")
        train_data = train_data[~train_data["month"].isin(purge_months)]

        train_data = train_data.dropna(subset=feature_cols + ["amount_new_house_transactions"])
        val_data = val_data.dropna(subset=feature_cols + ["amount_new_house_transactions"])
        if val_data.empty or train_data.empty:
            continue

        X_train = train_data[feature_cols]
        y_train = train_data["amount_new_house_transactions"]
        X_val = val_data[feature_cols]
        y_val = val_data["amount_new_house_transactions"].to_numpy()

        # Lag1 baseline
        lag_pred = val_data["lag_1"].to_numpy()
        score, d_ratio, ape_gt1 = _evaluate_predictions(y_val, lag_pred)
        records.append(
            {
                "fold": fold_name,
                "model": "Lag1",
                "score": score,
                "d_ratio": d_ratio,
                "ape_gt1_ratio": ape_gt1,
            }
        )

        # LightGBM baseline
        lgb_model = LGBMRegressor(
            objective="tweedie",
            tweedie_variance_power=1.2,
            random_state=config.random_seed,
        )
        lgb_model.fit(X_train, y_train)
        lgb_pred = lgb_model.predict(X_val)
        score, d_ratio, ape_gt1 = _evaluate_predictions(y_val, lgb_pred)
        records.append(
            {
                "fold": fold_name,
                "model": "LightGBM",
                "score": score,
                "d_ratio": d_ratio,
                "ape_gt1_ratio": ape_gt1,
            }
        )

    fold_results = pd.DataFrame(records)
    save_table_with_markdown(
        fold_results,
        config.reports_dir / "backtest" / "fold_scores.csv",
    )

    if not fold_results.empty:
        pivot = fold_results.pivot(index="fold", columns="model", values="score")
        fig, ax = plt.subplots(figsize=(8, 5))
        pivot.plot(marker="o", ax=ax)
        ax.set_ylabel("Score")
        ax.set_title("Baseline Scores by Fold")
        fig.savefig(config.reports_dir / "baseline" / "score_vs_fold.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    series = new_house.groupby("month")["amount_new_house_transactions"].sum().sort_index()
    sarima_train = series.loc[: "2024-06"]
    sarima_model = sm.tsa.statespace.SARIMAX(
        sarima_train,
        order=(1, 1, 1),
        seasonal_order=(0, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    sarima_forecast = sarima_model.get_forecast(steps=6)
    forecast_index = pd.date_range("2024-07-01", periods=6, freq="MS")
    sarima_df = pd.DataFrame(
        {
            "month": forecast_index,
            "prediction": sarima_forecast.predicted_mean,
        }
    )
    sarima_path = config.reports_dir / "baseline" / "sarima_forecast.csv"
    sarima_df.to_csv(sarima_path, index=False)

    if pd.Timestamp("2024-07-01") in series.index:
        actual = series.loc[pd.Timestamp("2024-07-01")]
        pred = sarima_forecast.predicted_mean.iloc[0]
        sarima_score, _, _ = _evaluate_predictions(np.array([actual]), np.array([pred]))
    else:
        sarima_score = float("nan")

    baseline_summary = [
        {
            "model": "LightGBM",
            "score": float(fold_results[fold_results["model"] == "LightGBM"]["score"].mean()),
        },
        {
            "model": "Lag1",
            "score": float(fold_results[fold_results["model"] == "Lag1"]["score"].mean()),
        },
        {
            "model": "SARIMA",
            "score": sarima_score,
        },
    ]
    baseline_df = pd.DataFrame(baseline_summary)
    save_table_with_markdown(
        baseline_df,
        config.reports_dir / "baseline" / "baseline_scores.csv",
    )

    summary_text = "\n".join(
        [
            "# Baseline Summary",
            "",
            "- LightGBM mean score: {:.4f}".format(baseline_summary[0]["score"]),
            "- Lag1 mean score: {:.4f}".format(baseline_summary[1]["score"]),
            "- SARIMA validation score (2024-07): {:.4f}".format(baseline_summary[2]["score"]),
        ]
    )
    (config.reports_dir / "baseline" / "baseline_summary.md").write_text(summary_text, encoding="utf-8")


def lock_environment(config: Config) -> None:
    """Freeze Python environment packages."""

    path = config.reports_dir / "environment" / "requirements_lock.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=f, check=True)


@app.command()
def main(
    headless: bool = typer.Option(
        True,
        "--headless",
        "--no-headless",
        help="Run Matplotlib in headless mode",
    ),
) -> None:
    """Entry point for running the EDA pipeline."""

    # 註解：定位專案根目錄
    base_dir = Path(__file__).resolve().parents[1]
    config = Config.create(base_dir)
    set_headless_backend(headless)
    configure_logging(config.reports_dir / "eda.log")
    logging.info("Starting EDA pipeline")
    validate_environment(config)
    clean_reports_dir(config)
    ensure_directories(config)
    ensure_data_release_log(config)
    ensure_font_available(config.font_name)
    set_global_seed(config.random_seed)
    sns.set_theme(style="whitegrid")
    verify_data_assets(config)
    datasets = load_datasets(config)
    create_sector_city_mapping(config, datasets)
    generate_target_trends(datasets["new_house_transactions"], config)
    generate_preowned_analysis(
        datasets["new_house_transactions"],
        datasets["pre_owned_house_transactions"],
        config,
    )
    generate_land_analysis(
        datasets["new_house_transactions"], datasets["land_transactions"], config
    )
    search_features = prepare_search_analysis(
        datasets["city_search_index"],
        datasets["new_house_transactions"],
        config,
    )
    process_macro_indicators(datasets["city_indexes"], datasets["new_house_transactions"], config)
    analyze_poi_features(datasets["sector_POI"], config)
    detect_structural_breaks(datasets["new_house_transactions"], config)
    analyze_target_decomposition(datasets["new_house_transactions"], config)
    run_baseline_models(
        datasets["new_house_transactions"],
        datasets["land_transactions"],
        search_features,
        config,
    )
    lock_environment(config)
    logging.info("EDA pipeline completed")


if __name__ == "__main__":
    app()
