from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features_v3 import build_feature_matrix_v3


RAW_COLUMNS = {
    "amount_new_house_transactions": 100.0,
    "num_new_house_transactions": 20.0,
    "area_new_house_transactions": 300.0,
    "price_new_house_transactions": 10.0,
    "area_per_unit_new_house_transactions": 15.0,
    "total_price_per_unit_new_house_transactions": 30.0,
    "num_new_house_available_for_sale": 25.0,
    "area_new_house_available_for_sale": 250.0,
    "period_new_house_sell_through": 2.0,
    "amount_pre_owned_house_transactions": 80.0,
    "num_pre_owned_house_transactions": 18.0,
    "area_pre_owned_house_transactions": 200.0,
    "price_pre_owned_house_transactions": 9.0,
    "transaction_amount": 50.0,
    "num_land_transactions": 2.0,
    "construction_area": 60.0,
    "planned_building_area": 45.0,
}


def _make_panel(tmp_path: Path) -> Path:
    months = pd.date_range("2024-05-01", periods=4, freq="MS")
    sectors = [1, 2]
    records = []
    for month in months:
        for sector in sectors:
            base_multiplier = 1.0 + 0.1 * sector + 0.05 * (month.month - 5)
            row = {
                "month": month,
                "sector_id": sector,
                "id": f"{month.strftime('%Y %b')}_sector {sector}",
                "target": 120.0 if month <= pd.Timestamp("2024-07-01") else np.nan,
                "target_filled": 120.0 if month <= pd.Timestamp("2024-07-01") else 0.0,
                "target_available_flag": 1 if month <= pd.Timestamp("2024-07-01") else 0,
                "target_filled_was_missing": 0,
                "is_future": 1 if month > pd.Timestamp("2024-07-01") else 0,
                "resident_population": 1000 * sector,
                "resident_population_dense": 2.0 * sector,
                "population_scale": 1.0 * sector,
                "population_scale_dense": 1.5 * sector,
                "office_population": 50.0 * sector,
                "office_population_dense": 0.5 * sector,
                "surrounding_housing_average_price": 30.0 * sector,
                "surrounding_shop_average_rent": 5.0 * sector,
                "sector_coverage": 0.8,
                "city_gdp_100m": 500.0,
                "city_secondary_industry_100m": 200.0,
                "city_tertiary_industry_100m": 250.0,
                "city_gdp_per_capita_yuan": 10000.0,
                "city_total_households_10k": 30.0,
                "city_year_end_resident_population_10k": 40.0,
                "city_total_retail_sales_of_consumer_goods_100m": 120.0,
                "city_per_capita_disposable_income_absolute_yuan": 40000.0,
                "city_annual_average_wage_urban_non_private_employees_yuan": 90000.0,
                "city_number_of_universities": 10.0,
                "city_hospital_beds_10k": 5.0,
                "city_number_of_operating_bus_lines": 150.0,
            }
            for key, value in RAW_COLUMNS.items():
                row[key] = value * base_multiplier
                row[f"{key}_nearby_sectors"] = value * (base_multiplier + 0.2)
            for flag in [
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
            ]:
                row[f"{flag}_was_interpolated"] = 0
            for idx in range(12):
                col = f"search_kw_kw{idx}"
                row[col] = (idx + 1) * base_multiplier
                row[f"{col}_was_missing"] = 0
            row["poi_custom_dense"] = 0.5 * sector
            records.append(row)
    panel = pd.DataFrame(records)
    panel_path = tmp_path / "panel.parquet"
    panel.to_parquet(panel_path, index=False)
    return panel_path


def test_feature_counts_and_categories(tmp_path: Path):
    panel_path = _make_panel(tmp_path)
    reports_dir = tmp_path / "reports"
    features_df, metadata = build_feature_matrix_v3(
        panel_path=panel_path,
        forecast_start="2024-08-01",
        features_path=None,
        reports_dir=reports_dir,
    )

    inventory = metadata["inventory"]
    assert 537 <= inventory["total_features"] <= 567
    assert "amount_new_house_transactions_log1p" in features_df.columns
    assert "amount_new_house_transactions_lag_1" in features_df.columns
    assert "amount_new_house_transactions_growth_1_log1p" in features_df.columns
    assert "amount_new_house_transactions_share_log1p" in features_df.columns
    assert "search_kw_kw0_pct_change_1" in features_df.columns
    assert "poi_pca_1" in features_df.columns
    assert "selected_search_keywords" in metadata
    assert len(metadata["selected_search_keywords"]) <= 12

    missing_report_path = reports_dir / "missing_value_report.json"
    assert missing_report_path.exists()
    with missing_report_path.open("r", encoding="utf-8") as handle:
        report_entries = json.load(handle)
    assert report_entries
    assert {entry["projection_stage"] for entry in report_entries}.issubset({"observed", "forecast"})

    inventory_path = reports_dir / "feature_inventory_v3.json"
    assert inventory_path.exists()
    with inventory_path.open("r", encoding="utf-8") as handle:
        inventory_payload = json.load(handle)
    assert "by_category" in inventory_payload
    assert "log1p" in inventory_payload["by_category"]
    assert "weighted_mean" in inventory_payload["by_category"]
