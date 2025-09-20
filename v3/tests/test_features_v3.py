import pandas as pd

from src.features_v2 import build_feature_matrix_v2

def _sample_panel():
    data = {
        "month": ["2024-05-01", "2024-06-01", "2024-07-01"],
        "sector": ["sector 1", "sector 1", "sector 1"],
        "sector_id": [1, 1, 1],
        "num_new_house_transactions": [1.0, None, None],
        "area_new_house_transactions": [10.0, 11.0, None],
        "price_new_house_transactions": [100.0, 99.0, None],
        "amount_new_house_transactions": [100.0, 110.0, 120.0],
        "area_per_unit_new_house_transactions": [5.0, None, None],
        "total_price_per_unit_new_house_transactions": [50.0, 55.0, None],
        "num_new_house_available_for_sale": [2.0, 2.0, None],
        "area_new_house_available_for_sale": [20.0, 21.0, None],
        "period_new_house_sell_through": [1.0, None, None],
        "resident_population": [1000, 1000, 1000],
        "population_scale": [1.0, 1.0, 1.0],
        "search_kw_housing": [100.0, 110.0, None],
        "search_kw_housing_was_missing": [0, 0, 1],
        "city_gdp": [1000.0, 1010.0, 1020.0],
        "city_gdp_was_interpolated": [0, 0, 1],
        "target": [100.0, 110.0, 120.0],
        "target_filled": [100.0, 110.0, 120.0],
        "is_future": [False, False, False],
    }
    return pd.DataFrame(data)


def test_missing_value_policy_adds_flags_and_imputes():
    panel_df = _sample_panel()
    features_df, metadata = build_feature_matrix_v2(panel_df, features_path=None, reports_dir=None)

    assert "num_new_house_transactions_was_missing" in features_df.columns
    assert not features_df["num_new_house_transactions"].isna().any()
    assert "num_new_house_transactions_share" in metadata.get("share_features", [])
    assert "num_new_house_transactions_log1p" in metadata.get("log1p_features", [])
