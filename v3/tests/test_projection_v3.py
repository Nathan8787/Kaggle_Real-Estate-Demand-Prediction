from __future__ import annotations

import pandas as pd
import pytest

from src.projection_v3 import apply_projection


def _make_panel() -> pd.DataFrame:
    records = []
    months = [
        pd.Timestamp("2023-08-01"),
        pd.Timestamp("2024-05-01"),
        pd.Timestamp("2024-06-01"),
        pd.Timestamp("2024-07-01"),
        pd.Timestamp("2024-08-01"),
        pd.Timestamp("2024-09-01"),
    ]
    for month in months:
        base_value = 10 + (month.month % 5)
        records.append(
            {
                "month": month,
                "sector_id": 1,
                "amount_new_house_transactions": float(base_value),
                "search_kw_alpha": 5.0,
            }
        )
    # sector 2 with limited history (only two observations)
    records.append(
        {
            "month": pd.Timestamp("2023-08-01"),
            "sector_id": 2,
            "amount_new_house_transactions": 9.0,
            "search_kw_alpha": 2.0,
        }
    )
    records.append(
        {
            "month": pd.Timestamp("2024-05-01"),
            "sector_id": 2,
            "amount_new_house_transactions": 8.0,
            "search_kw_alpha": 2.0,
        }
    )
    records.append(
        {
            "month": pd.Timestamp("2024-08-01"),
            "sector_id": 2,
            "amount_new_house_transactions": float("nan"),
            "search_kw_alpha": float("nan"),
        }
    )
    panel = pd.DataFrame(records).sort_values(["sector_id", "month"]).reset_index(drop=True)
    return panel


def test_projection_respects_cutoff():
    panel = _make_panel()
    projected = apply_projection(panel, pd.Timestamp("2024-08-01"))

    # Sector 1 has four historical values -> median fallback (source code 2)
    sector1_aug = projected[
        (projected["sector_id"] == 1) & (projected["month"] == pd.Timestamp("2024-08-01"))
    ]
    assert sector1_aug["amount_new_house_transactions"].iloc[0] == pytest.approx(11.5, rel=1e-6)
    assert sector1_aug["amount_new_house_transactions_proj_source"].iloc[0] == 2
    assert sector1_aug["amount_new_house_transactions_proj_overridden"].iloc[0] == 1

    # Future month (September) should also be projected and not reuse the manual override
    sector1_sep = projected[
        (projected["sector_id"] == 1) & (projected["month"] == pd.Timestamp("2024-09-01"))
    ]
    assert sector1_sep["amount_new_house_transactions"].iloc[0] == pytest.approx(11.5, rel=1e-6)
    assert sector1_sep["amount_new_house_transactions_proj_source"].iloc[0] == 2

    # Sector 2 has insufficient history and should borrow August median from other sectors (source code 3)
    sector2_aug = projected[
        (projected["sector_id"] == 2) & (projected["month"] == pd.Timestamp("2024-08-01"))
    ]
    assert sector2_aug["amount_new_house_transactions"].iloc[0] == pytest.approx(11.0, rel=1e-6)
    assert sector2_aug["amount_new_house_transactions_proj_source"].iloc[0] == 3

    # Observed months remain untouched with source flag 0
    sector1_july = projected[
        (projected["sector_id"] == 1) & (projected["month"] == pd.Timestamp("2024-07-01"))
    ]
    assert sector1_july["amount_new_house_transactions_proj_source"].iloc[0] == 0
