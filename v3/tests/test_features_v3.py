from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features_v3 import build_feature_matrix_v3  # noqa: E402
from src.panel_builder_v3 import (  # noqa: E402
    attach_target_v3,
    build_calendar_v3,
    load_raw_tables_v3,
    merge_sources_v3,
)


def _build_panel(tmp_path: Path) -> Path:
    config_path = ROOT / "config" / "data_paths_v3.yaml"
    tables = load_raw_tables_v3(config_path)
    calendar = build_calendar_v3(tables)
    merged = merge_sources_v3(calendar, tables)
    panel = attach_target_v3(merged)
    panel_dir = tmp_path / "data" / "processed"
    panel_dir.mkdir(parents=True, exist_ok=True)
    panel_path = panel_dir / "panel_v3.parquet"
    panel.to_parquet(panel_path, index=False)
    return panel_path


def test_feature_counts_and_categories(tmp_path: Path):
    panel_path = _build_panel(tmp_path)
    reports_dir = tmp_path / "reports_v3"
    features_df, metadata = build_feature_matrix_v3(
        panel_path=panel_path,
        forecast_start="2024-08-01",
        features_path=None,
        reports_dir=reports_dir,
    )

    inventory = metadata["inventory"]
    assert 450 <= inventory["total_features"] <= 650
    assert inventory["projection_guard"]["status"] in {"ok", "warning"}

    missing_report_path = reports_dir / "missing_value_report.json"
    assert missing_report_path.exists()
    with missing_report_path.open("r", encoding="utf-8") as handle:
        report_entries = json.load(handle)
    assert report_entries
    assert {entry["projection_stage"] for entry in report_entries}.issubset({"observed", "forecast"})

    search_report_path = reports_dir / "search_keywords_v3.json"
    assert search_report_path.exists()
    with search_report_path.open("r", encoding="utf-8") as handle:
        search_payload = json.load(handle)
    assert len(search_payload.get("selected", [])) <= 12

    poi_summary_path = reports_dir / "poi_pca_summary.json"
    assert poi_summary_path.exists()

    inventory_path = reports_dir / "feature_inventory_v3.json"
    assert inventory_path.exists()
    with inventory_path.open("r", encoding="utf-8") as handle:
        inventory_payload = json.load(handle)
    assert "by_category" in inventory_payload
    assert "log1p" in inventory_payload["by_category"]
    assert "weighted_mean" in inventory_payload["by_category"]
