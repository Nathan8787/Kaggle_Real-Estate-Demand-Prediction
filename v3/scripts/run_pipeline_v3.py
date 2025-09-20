from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

sys.path.append(str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

from src.automl_runner_v3 import run_training_pipeline
from src.features_v3 import build_feature_matrix_v3
from src.metrics_v3 import competition_score
from src.panel_builder_v3 import (
    attach_target_v3,
    build_calendar_v3,
    load_raw_tables_v3,
    merge_sources_v3,
    save_panel_v3,
)
from src.predict_v3 import generate_submission
from src.splits_v3 import HOLDOUT_WINDOW


def _relocate_panel_summary(panel_path: Path, reports_dir: Path) -> None:
    """Ensure the panel summary JSON is located under the requested reports directory."""

    default_reports_dir = panel_path.parent.parent / "reports_v3"
    default_report_path = default_reports_dir / "panel_build_summary.json"
    target_report_path = reports_dir / "panel_build_summary.json"

    if not default_report_path.exists():
        return

    reports_dir.mkdir(parents=True, exist_ok=True)
    if default_report_path.resolve() == target_report_path.resolve():
        return

    target_report_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(default_report_path), target_report_path)
    try:
        if not any(default_reports_dir.iterdir()):
            default_reports_dir.rmdir()
    except OSError:
        # Directory still contains other files; leave it in place.
        pass


def _load_scaler_artifact(scaler_path: Path) -> tuple[object | None, list[str]]:
    payload = joblib.load(scaler_path)
    if isinstance(payload, dict):
        scaler = payload.get("scaler")
        float_cols = list(payload.get("float_cols", []))
    else:
        scaler = payload
        float_cols: list[str] = []
    return scaler, float_cols


def _compute_holdout_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = y_pred - y_true
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    mape = float(np.mean(np.abs(error) / (np.abs(y_true) + 1e-6)))
    smape = float(
        np.mean(2.0 * np.abs(error) / (np.abs(y_true) + np.abs(y_pred) + 1e-6))
    )
    score = float(competition_score(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if y_true.size > 1 else 0.0
    return {
        "competition_score": score,
        "SMAPE": smape,
        "MAPE": mape,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


def build_panel(args: argparse.Namespace) -> None:
    reports_dir = Path(args.reports_dir)
    tables = load_raw_tables_v3(Path(args.data_config))
    calendar = build_calendar_v3(tables)
    panel = merge_sources_v3(calendar, tables)
    panel = attach_target_v3(panel)
    panel_path = Path(args.panel_path)
    save_panel_v3(panel, panel_path)
    _relocate_panel_summary(panel_path, reports_dir)
    print(f"Panel saved to {panel_path}")


def build_features(args: argparse.Namespace) -> None:
    panel_path = Path(args.panel_path)
    if not panel_path.exists():
        raise FileNotFoundError(f"Panel file not found: {panel_path}")

    features_path = Path(args.features_path)
    reports_dir = Path(args.reports_dir)
    build_feature_matrix_v3(
        panel_path=panel_path,
        forecast_start=args.forecast_start,
        features_path=features_path,
        reports_dir=reports_dir,
    )
    print(f"Features saved to {features_path}")


def run_xgb(args: argparse.Namespace) -> None:
    features_path = Path(args.features_path)
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    features_df = pd.read_parquet(features_path)
    summary = run_training_pipeline(
        features_df,
        config_path=Path(args.config),
        models_dir=Path(args.models_dir),
        reports_dir=Path(args.reports_dir),
        logs_dir=Path(args.logs_dir),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def evaluate_holdout(args: argparse.Namespace) -> None:
    features_path = Path(args.features_path)
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    features_df = pd.read_parquet(features_path)
    features_df["month"] = pd.to_datetime(features_df["month"], errors="coerce")

    _, _, holdout_start, holdout_end = HOLDOUT_WINDOW
    holdout_start_ts = pd.Timestamp(holdout_start)
    holdout_end_ts = pd.Timestamp(holdout_end)

    mask = (
        (features_df["month"] >= holdout_start_ts)
        & (features_df["month"] <= holdout_end_ts)
        & features_df["target"].notna()
    )
    holdout_slice = features_df.loc[mask].copy()
    if holdout_slice.empty:
        raise ValueError("Holdout window does not contain labeled observations")

    feature_cols = [
        col for col in features_df.columns if col not in {"month", "id", "target"}
    ]
    X_holdout = holdout_slice[feature_cols].copy()
    y_true = holdout_slice["target"].astype(float).to_numpy()

    scaler, float_cols = _load_scaler_artifact(Path(args.scaler_path))
    if scaler is not None and float_cols:
        X_holdout.loc[:, float_cols] = scaler.transform(X_holdout[float_cols])

    model = XGBRegressor()
    model.load_model(args.model_path)
    preds_log = model.predict(X_holdout.values)
    preds = np.clip(np.expm1(preds_log), 0.0, None)

    metrics = _compute_holdout_metrics(y_true, preds)

    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "metrics": metrics,
        "projection_drift_report": str(
            reports_dir / "projection_drift_202407.json"
        ),
    }
    with (reports_dir / "holdout_metrics_v3.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    print(json.dumps(payload, indent=2, ensure_ascii=False))


def predict(args: argparse.Namespace) -> None:
    submission_path = generate_submission(
        panel_path=args.panel_path,
        features_path=args.features_path,
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        sample_submission=args.sample_submission,
        output_path=args.output_path,
        reports_dir=args.reports_dir,
        residual_model=args.residual_model,
        residual_scaler=args.residual_scaler,
        residual_features=args.residual_features,
    )
    print(f"Submission written to {submission_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline runner for v3 AutoML workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    panel_parser = subparsers.add_parser("build_panel")
    panel_parser.add_argument("--data-config", default="v3/config/data_paths_v3.yaml")
    panel_parser.add_argument("--panel-path", default="data/processed/panel_v3.parquet")
    panel_parser.add_argument("--reports-dir", default="reports_v3")
    panel_parser.set_defaults(func=build_panel)

    features_parser = subparsers.add_parser("build_features")
    features_parser.add_argument("--panel-path", default="data/processed/panel_v3.parquet")
    features_parser.add_argument("--features-path", default="data/processed/features_v3.parquet")
    features_parser.add_argument("--reports-dir", default="reports_v3")
    features_parser.add_argument("--forecast-start", required=True)
    features_parser.set_defaults(func=build_features)

    automl_parser = subparsers.add_parser("run_xgb")
    automl_parser.add_argument("--features-path", default="data/processed/features_v3.parquet")
    automl_parser.add_argument("--config", default="v3/config/automl_v3.yaml")
    automl_parser.add_argument("--models-dir", default="models_v3")
    automl_parser.add_argument("--reports-dir", default="reports_v3")
    automl_parser.add_argument("--logs-dir", default="logs_v3")
    automl_parser.set_defaults(func=run_xgb)

    holdout_parser = subparsers.add_parser("evaluate_holdout")
    holdout_parser.add_argument("--features-path", default="data/processed/features_v3.parquet")
    holdout_parser.add_argument("--model-path", default="models_v3/best_model_holdout.json")
    holdout_parser.add_argument("--scaler-path", default="models_v3/scaler_holdout.pkl")
    holdout_parser.add_argument("--reports-dir", default="reports_v3")
    holdout_parser.set_defaults(func=evaluate_holdout)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--panel-path", default="data/processed/panel_v3.parquet")
    predict_parser.add_argument("--features-path", default="data/processed/features_v3.parquet")
    predict_parser.add_argument("--model-path", default="models_v3/best_model_full.json")
    predict_parser.add_argument("--scaler-path", default="models_v3/scaler_full.pkl")
    predict_parser.add_argument("--sample-submission", default="../sample_submission.csv")
    predict_parser.add_argument("--output-path", default="submission_v3/submission.csv")
    predict_parser.add_argument("--reports-dir", default="reports_v3")
    predict_parser.add_argument("--residual-model")
    predict_parser.add_argument("--residual-scaler")
    predict_parser.add_argument("--residual-features")
    predict_parser.set_defaults(func=predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
