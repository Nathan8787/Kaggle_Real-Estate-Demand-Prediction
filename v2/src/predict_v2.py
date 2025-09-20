from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd

from .features_v2 import build_feature_matrix_v2

__all__ = ["generate_predictions"]

_EXCLUDE_COLUMNS = {"id", "sector", "month", "target", "target_filled", "is_future"}


def _load_feature_columns(features_path: Path) -> list[str]:
    features_df = pd.read_parquet(features_path)
    return [col for col in features_df.columns if col not in _EXCLUDE_COLUMNS]


def _select_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    working = df.copy()
    missing = [col for col in columns if col not in working.columns]
    for col in missing:
        working[col] = 0.0
    return working.loc[:, list(columns)].astype(float)


def _load_residual_artifacts(
    model_path: Path | None,
    scaler_path: Path | None,
    features_path: Path | None,
) -> tuple[object | None, object | None, list[str] | None]:
    if not (model_path and scaler_path and features_path):
        return None, None, None

    residual_model = joblib.load(model_path)
    residual_scaler = joblib.load(scaler_path)
    with features_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if isinstance(payload, dict):
        features = payload.get("features")
    else:
        features = payload
    if not isinstance(features, list):
        raise ValueError("Residual features file must contain a list of feature names")
    return residual_model, residual_scaler, features


def generate_predictions(
    panel_path: str | Path,
    features_path: str | Path,
    model_path: str | Path,
    scaler_path: str | Path,
    sample_submission_path: str | Path,
    output_path: str | Path,
    residual_model_path: str | Path | None = None,
    residual_scaler_path: str | Path | None = None,
    residual_features_path: str | Path | None = None,
) -> dict:
    panel_path = Path(panel_path)
    features_path = Path(features_path)
    model_path = Path(model_path)
    scaler_path = Path(scaler_path)
    sample_submission_path = Path(sample_submission_path)
    output_path = Path(output_path)

    if not panel_path.exists():
        raise FileNotFoundError(f"Panel file not found: {panel_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")
    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError("Model and scaler artifacts are required before prediction")
    if not sample_submission_path.exists():
        raise FileNotFoundError(f"Sample submission not found: {sample_submission_path}")

    working_panel = pd.read_parquet(panel_path).copy()
    working_panel["month"] = pd.to_datetime(working_panel["month"], errors="coerce")

    feature_columns = _load_feature_columns(features_path)
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)

    residual_model, residual_scaler, residual_features = _load_residual_artifacts(
        Path(residual_model_path) if residual_model_path else None,
        Path(residual_scaler_path) if residual_scaler_path else None,
        Path(residual_features_path) if residual_features_path else None,
    )

    forecast_months = sorted(
        pd.to_datetime(
            working_panel.loc[working_panel["is_future"], "month"].dropna().unique()
        )
    )
    if not forecast_months:
        raise ValueError("No future months detected in the panel for prediction")

    predictions: list[pd.DataFrame] = []

    original_ids = working_panel["id"].copy()

    for month in forecast_months:
        features_df, _ = build_feature_matrix_v2(working_panel, features_path=None, reports_dir=None)
        current_subset = features_df[features_df["month"] == month].copy()
        if current_subset.empty:
            raise ValueError(f"No features generated for month {month}")

        feature_matrix = _select_columns(current_subset, feature_columns)
        transformed = scaler.transform(feature_matrix)
        preds_log = model.predict(transformed)
        preds = np.clip(np.expm1(preds_log), 0.0, None)

        if residual_model is not None and residual_scaler is not None and residual_features is not None:
            residual_frame = current_subset.copy()
            residual_frame["prediction"] = preds
            residual_inputs = _select_columns(residual_frame, residual_features)
            residual_adjustments = residual_model.predict(residual_scaler.transform(residual_inputs))
            preds = np.clip(preds + residual_adjustments, 0.0, None)

        prediction_frame = pd.DataFrame(
            {
                "id": current_subset["id"].values,
                "new_house_transaction_amount": preds,
            }
        )
        predictions.append(prediction_frame)

        predictions_series = pd.Series(preds, index=current_subset["id"].values)
        panel_indexed = working_panel.set_index("id")
        panel_indexed.loc[predictions_series.index, "target_filled"] = predictions_series.values
        panel_indexed.loc[predictions_series.index, "target"] = predictions_series.values
        working_panel = panel_indexed.loc[original_ids].reset_index()

    predictions_df = pd.concat(predictions, ignore_index=True)
    sample = pd.read_csv(sample_submission_path)
    submission = sample[["id"]].merge(predictions_df, on="id", how="left")
    if submission["new_house_transaction_amount"].isna().any():
        missing = submission.loc[submission["new_house_transaction_amount"].isna(), "id"].tolist()
        raise ValueError(f"Missing predictions for ids: {missing}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    summary = {
        "prediction_count": int(len(submission)),
        "prediction_mean": float(submission["new_house_transaction_amount"].mean()),
        "prediction_std": float(submission["new_house_transaction_amount"].std()),
        "prediction_min": float(submission["new_house_transaction_amount"].min()),
        "prediction_max": float(submission["new_house_transaction_amount"].max()),
    }

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    with (reports_dir / "prediction_summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    return summary
