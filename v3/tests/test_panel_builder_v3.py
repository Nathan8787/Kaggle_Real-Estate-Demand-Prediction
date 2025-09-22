from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.panel_builder_v3 import (  # noqa: E402
    attach_target_v3,
    build_calendar_v3,
    load_raw_tables_v3,
    merge_sources_v3,
    save_panel_v3,
)


def test_panel_builder_outputs_summary(tmp_path: Path):
    config_path = ROOT / "config" / "data_paths_v3.yaml"
    tables = load_raw_tables_v3(config_path)
    calendar = build_calendar_v3(tables)
    merged = merge_sources_v3(calendar, tables)
    panel = attach_target_v3(merged)

    assert "city_id" in panel.columns
    assert panel["city_id"].dtype == np.int16
    assert panel["sector_id"].dtype == np.int32
    assert set(panel["target_available_flag"].unique()).issubset({0, 1})

    panel_path = tmp_path / "data" / "processed" / "panel_v3.parquet"
    with patch.object(pd.DataFrame, "to_parquet", return_value=None) as mock_to_parquet:
        save_panel_v3(panel, panel_path)
        assert mock_to_parquet.called

    summary_path = tmp_path / "data" / "reports_v3" / "panel_build_summary.json"
    assert summary_path.exists()
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["sector_count"] > 0
    assert summary["labeled_rows"] > 0
    assert summary["future_rows"] > 0
