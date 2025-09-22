from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.panel_builder_v3 import (  # noqa: E402
    attach_target_v3,
    build_calendar_v3,
    load_raw_tables_v3,
    merge_sources_v3,
)
from src.projection_v3 import PROJECTION_COLUMNS, apply_projection  # noqa: E402


def _build_real_panel() -> pd.DataFrame:
    config_path = ROOT / "config" / "data_paths_v3.yaml"
    tables = load_raw_tables_v3(config_path)
    calendar = build_calendar_v3(tables)
    merged = merge_sources_v3(calendar, tables)
    panel = attach_target_v3(merged)
    return panel


def test_projection_respects_cutoff():
    panel = _build_real_panel()
    forecast_start = pd.Timestamp("2024-08-01")
    projected = apply_projection(panel, forecast_start)

    future_mask = projected["month"] >= forecast_start
    past_mask = ~future_mask

    projected_columns = [col for col in PROJECTION_COLUMNS if col in projected.columns]
    assert projected_columns, "expected projection columns in panel"

    for column in projected_columns:
        source_col = f"{column}_proj_source"
        assert source_col in projected.columns
        # Observed months remain flagged as 0
        assert set(projected.loc[past_mask, source_col].unique()) <= {0}
        # Future months should not contain NaN projections
        assert projected.loc[future_mask, column].notna().all()

    # Source codes limited to the defined range
    source_values = []
    for column in projected_columns:
        source_values.append(projected[f"{column}_proj_source"].to_numpy())
    source_values = np.concatenate(source_values)
    assert set(np.unique(source_values)).issubset({0, 1, 2, 3, 4})

    # At least one column should fall back to shared or global sources in the forecast horizon
    fallback_triggered = False
    for column in projected_columns:
        codes = projected.loc[future_mask, f"{column}_proj_source"].to_numpy()
        if np.isin(codes, [3, 4]).any():
            fallback_triggered = True
            break
    assert fallback_triggered
