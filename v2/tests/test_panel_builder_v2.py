import pandas as pd

from src.panel_builder_v2 import build_calendar, merge_sources


def _base_month_df(months, sectors, value_map):
    records = []
    for month in months:
        for sector in sectors:
            row = {"month": month.strftime("%Y-%m-%d"), "sector": sector}
            row.update(value_map)
            records.append(row)
    return pd.DataFrame(records)


def test_extend_monthly_tables_fills_future_months():
    sectors = ["sector 1", "sector 2"]
    base_months = [pd.Timestamp("2024-05-01"), pd.Timestamp("2024-06-01"), pd.Timestamp("2024-07-01")]

    new_house_values = {
        "num_new_house_transactions": 5.0,
        "area_new_house_transactions": 50.0,
        "price_new_house_transactions": 1000.0,
        "amount_new_house_transactions": 5000.0,
        "area_per_unit_new_house_transactions": 10.0,
        "total_price_per_unit_new_house_transactions": 100.0,
        "num_new_house_available_for_sale": 5.0,
        "area_new_house_available_for_sale": 40.0,
        "period_new_house_sell_through": 2.0,
    }
    new_house = _base_month_df(base_months, sectors, new_house_values)

    new_house_nearby = new_house.rename(columns={
        col: f"{col}_nearby" for col in new_house.columns if col not in {"month", "sector"}
    })

    pre_owned = new_house[["month", "sector"]].copy()
    pre_owned["num_pre_owned_house_transactions"] = 2.0
    pre_owned["area_pre_owned_house_transactions"] = 20.0
    pre_owned["price_pre_owned_house_transactions"] = 800.0
    pre_owned["amount_pre_owned_house_transactions"] = 1600.0
    pre_owned_nearby = pre_owned.rename(columns={
        col: f"{col}_nearby" for col in pre_owned.columns if col not in {"month", "sector"}
    })

    land = new_house[["month", "sector"]].copy()
    land["num_land_transactions"] = 1.0
    land["construction_area"] = 20.0
    land["planned_building_area"] = 30.0
    land["transaction_amount"] = 40.0
    land_nearby = land.rename(columns={
        col: f"{col}_nearby" for col in land.columns if col not in {"month", "sector"}
    })

    sector_poi = pd.DataFrame({
        "sector": sectors,
        "resident_population": [1000, 2000],
        "population_scale": [1.0, 1.0],
        "poi_dense": [0.1, 0.2],
    })

    city_search_index = pd.DataFrame({
        "month": ["2024-05-01", "2024-06-01", "2024-07-01"],
        "keyword": ["房價", "房價", "房價"],
        "keyword_slug": ["fang_jia", "fang_jia", "fang_jia"],
        "search_volume": [100, 110, 120],
        "source": ["google", "google", "google"],
    })

    city_indexes = pd.DataFrame({
        "city_indicator_data_year": [2021, 2022],
        "gdp": [1000, 1100],
    })

    raw_tables = {
        "new_house_transactions": new_house,
        "new_house_transactions_nearby_sectors": new_house_nearby,
        "pre_owned_house_transactions": pre_owned,
        "pre_owned_house_transactions_nearby_sectors": pre_owned_nearby,
        "land_transactions": land,
        "land_transactions_nearby_sectors": land_nearby,
        "sector_poi": sector_poi,
        "city_search_index": city_search_index,
        "city_indexes": city_indexes,
    }

    calendar = build_calendar({"new_house_transactions": new_house})
    panel = merge_sources(calendar, raw_tables)
    future = panel[panel["month"] > pd.Timestamp("2024-07-01")]
    assert not future["num_new_house_transactions"].isna().any()
    assert not future["num_land_transactions"].isna().any()
    search_cols = [col for col in panel.columns if col.startswith("search_kw_") and not col.endswith("_was_missing")]
    assert search_cols
    for col in search_cols:
        assert not panel[col].isna().any()
        flag_col = f"{col}_was_missing"
        assert flag_col in panel.columns
        unique_vals = set(panel[flag_col].dropna().unique())
        assert unique_vals.issubset({0, 1})
