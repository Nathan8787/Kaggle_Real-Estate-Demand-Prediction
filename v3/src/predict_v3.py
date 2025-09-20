from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from .features_v3 import build_feature_matrix_v3
from .panel_builder_v3 import FORECAST_MONTHS
from .projection_v3 import apply_projection

__all__ = ["generate_submission"]


def _load_residual_artifacts(
    residual_model_path: Optional[Path],
    residual_scaler_path: Optional[Path],
    residual_features_path: Optional[Path],
) -> Tuple[Optional[object], Optional[object], Optional[List[str]]]:
    if not (residual_model_path and residual_scaler_path and residual_features_path):
        return None, None, None
    model = joblib.load(residual_model_path)
    scaler = joblib.load(residual_scaler_path)
    with residual_features_path.open("r", encoding="utf-8") as handle:
        features = json.load(handle)
    if not isinstance(features, list):
        raise ValueError("Residual features specification must be a list")
    return model, scaler, features


def _prepare_feature_columns(training_features_path: Path) -> List[str]:
    training_df = pd.read_parquet(training_features_path)
    exclude = {"month", "id", "target"}
    return [col for col in training_df.columns if col not in exclude]


def _align_feature_matrix(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    aligned = pd.DataFrame(index=df.index)
    for column in columns:
        aligned[column] = df.get(column, 0.0)
    return aligned


def _apply_scaler_to_features(df: pd.DataFrame, scaler, float_cols: Sequence[str]) -> pd.DataFrame:
    if not float_cols:
        return df
    transformed = df.copy()
    transformed.loc[:, float_cols] = scaler.transform(df[float_cols])
    return transformed


def _plot_placeholder(output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title(title)
    ax.text(0.5, 0.5, "Plot placeholder", ha="center", va="center", fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def generate_submission(
    panel_path: Path | str,
    features_path: Path | str,
    model_path: Path | str,
    scaler_path: Path | str,
    sample_submission: Path | str,
    output_path: Path | str,
    reports_dir: Path | str = Path("reports_v3"),
    residual_model: Path | str | None = None,
    residual_scaler: Path | str | None = None,
    residual_features: Path | str | None = None,
) -> Path:
    panel_path = Path(panel_path)
    features_path = Path(features_path)
    model_path = Path(model_path)
    scaler_path = Path(scaler_path)
    sample_submission = Path(sample_submission)
    output_path = Path(output_path)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not panel_path.exists():
        raise FileNotFoundError(f"Panel data not found: {panel_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Training features not found: {features_path}")
    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError("Model and scaler artifacts are required for inference")

    working_panel = pd.read_parquet(panel_path)
    working_panel["month"] = pd.to_datetime(working_panel["month"], errors="coerce")

    feature_columns = _prepare_feature_columns(features_path)
    scaler_payload = joblib.load(scaler_path)
    scaler = scaler_payload.get("scaler") if isinstance(scaler_payload, dict) else scaler_payload
    float_cols = scaler_payload.get("float_cols") if isinstance(scaler_payload, dict) else []
    float_cols = list(float_cols) if float_cols else []

    model = XGBRegressor()
    model.load_model(model_path)

    residual_model_obj, residual_scaler_obj, residual_feature_names = _load_residual_artifacts(
        Path(residual_model) if residual_model else None,
        Path(residual_scaler) if residual_scaler else None,
        Path(residual_features) if residual_features else None,
    )

    predictions: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, object]] = []

    for forecast_month in FORECAST_MONTHS:
        working_panel = apply_projection(working_panel, forecast_month)
        projection_mask = working_panel["month"] == forecast_month
        projection_sources = working_panel.loc[projection_mask, "amount_new_house_transactions_proj_source"].to_numpy()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_panel_path = Path(tmpdir) / "panel_snapshot.parquet"
            working_panel.to_parquet(tmp_panel_path, index=False)
            features_df, _ = build_feature_matrix_v3(
                panel_path=tmp_panel_path,
                forecast_start=forecast_month,
                features_path=None,
                reports_dir=None,
            )

        current = features_df[features_df["month"] == forecast_month].copy()
        if current.empty:
            raise ValueError(f"No features generated for forecast month {forecast_month}")

        feature_matrix = _align_feature_matrix(current, feature_columns)
        feature_matrix = _apply_scaler_to_features(feature_matrix, scaler, float_cols)
        preds_log = model.predict(feature_matrix.values)
        preds = np.clip(np.expm1(preds_log), 0.0, None)

        if residual_model_obj is not None and residual_scaler_obj is not None and residual_feature_names is not None:
            residual_frame = current.copy()
            residual_frame["prediction"] = preds
            residual_inputs = _align_feature_matrix(residual_frame, residual_feature_names)
            residual_inputs = residual_scaler_obj.transform(residual_inputs)
            adjustments = residual_model_obj.predict(residual_inputs)
            preds = np.clip(preds + adjustments, 0.0, None)

        current_ids = current["id"].astype(str).values
        prediction_frame = pd.DataFrame(
            {"id": current_ids, "new_house_transaction_amount": preds, "month": forecast_month}
        )
        predictions.append(prediction_frame)

        update_mask = working_panel["id"].astype(str).isin(current_ids)
        working_panel.loc[update_mask, "target_filled"] = preds

        summary_rows.append(
            {
                "month": forecast_month.strftime("%Y-%m"),
                "mean": float(np.mean(preds)),
                "std": float(np.std(preds)),
                "min": float(np.min(preds)),
                "max": float(np.max(preds)),
                "projection_dependence": float(np.mean(projection_sources > 0)),
            }
        )

    predictions_df = pd.concat(predictions, ignore_index=True)
    sample_df = pd.read_csv(sample_submission)
    submission = sample_df.merge(predictions_df[["id", "new_house_transaction_amount"]], on="id", how="left")
    if submission["new_house_transaction_amount"].isna().any():
        missing_ids = submission.loc[submission["new_house_transaction_amount"].isna(), "id"].tolist()
        raise ValueError(f"Missing predictions for ids: {missing_ids}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    summary_df = pd.DataFrame(summary_rows)
    high_dependence = bool((summary_df["projection_dependence"] > 0.85).any())
    test_summary = {
        "monthly_summary": summary_rows,
        "high_projection_dependence": high_dependence,
    }
    with (reports_dir / "test_window_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(test_summary, handle, indent=2, ensure_ascii=False)

    _plot_placeholder(reports_dir / "plots" / "residual_vs_target_2024-07.png", "Residual vs Target 2024-07")
    _plot_placeholder(reports_dir / "plots" / "prediction_distribution_train.png", "Prediction Distribution")

    return output_path
