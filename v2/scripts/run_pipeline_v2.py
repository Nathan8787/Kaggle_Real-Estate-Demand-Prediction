from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.panel_builder_v2 import (
    load_raw_tables_v2,
    build_calendar,
    merge_sources,
    attach_target_v2,
    save_panel,
)
from src.features_v2 import build_feature_matrix_v2
from src.predict_v2 import generate_predictions
from src.automl_runner_v2 import run
from walk_forward_train import walk_forward


def build_panel(args: argparse.Namespace) -> None:
    tables = load_raw_tables_v2(args.data_config)
    calendar = build_calendar(tables)
    panel = merge_sources(calendar, tables)
    panel = attach_target_v2(panel)
    save_panel(panel, args.panel_path)
    print(f"Panel saved to {args.panel_path}")


def build_features(args: argparse.Namespace) -> None:
    panel_path = Path(args.panel_path)
    if not panel_path.exists():
        raise FileNotFoundError(f"Panel file not found: {panel_path}")
    panel_df = pd.read_parquet(panel_path)
    build_feature_matrix_v2(
        panel_df,
        features_path=args.features_path,
        reports_dir=args.reports_dir,
    )
    print(f"Features saved to {args.features_path}")


def run_xgb(args: argparse.Namespace) -> None:
    features_path = Path(args.features_path)
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")
    features_df = pd.read_parquet(features_path)
    summary = run(features_df, args.config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def predict(args: argparse.Namespace) -> None:
    summary = generate_predictions(
        panel_path=args.panel_path,
        features_path=args.features_path,
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        sample_submission_path=args.sample_submission,
        output_path=args.output_path,
        residual_model_path=args.residual_model,
        residual_scaler_path=args.residual_scaler,
        residual_features_path=args.residual_features,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Submission written to {args.output_path}")

def walk_forward_train(args: argparse.Namespace) -> None:
    walk_forward(args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline runner for v2 AutoML workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    panel_parser = subparsers.add_parser("build_panel")
    panel_parser.add_argument("--data-config", default="config/data_paths_v2.yaml")
    panel_parser.add_argument("--panel-path", default="data/processed/panel_v2.parquet")
    panel_parser.set_defaults(func=build_panel)

    features_parser = subparsers.add_parser("build_features")
    features_parser.add_argument("--panel-path", default="data/processed/panel_v2.parquet")
    features_parser.add_argument("--features-path", default="data/processed/features_v2.parquet")
    features_parser.add_argument("--reports-dir", default="reports")
    features_parser.set_defaults(func=build_features)

    automl_parser = subparsers.add_parser("run_xgb")
    automl_parser.add_argument("--features-path", default="data/processed/features_v2.parquet")
    automl_parser.add_argument("--config", default="config/automl_v2.yaml")
    automl_parser.set_defaults(func=run_xgb)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--panel-path", default="data/processed/panel_v2.parquet")
    predict_parser.add_argument("--features-path", default="data/processed/features_v2.parquet")
    predict_parser.add_argument("--model-path", default="models/best_model.pkl")
    predict_parser.add_argument("--scaler-path", default="models/scaler.pkl")
    predict_parser.add_argument("--sample-submission", default="../sample_submission.csv")
    predict_parser.add_argument("--output-path", default="submission/submission.csv")
    predict_parser.add_argument("--residual-model")
    predict_parser.add_argument("--residual-scaler")
    predict_parser.add_argument("--residual-features")
    predict_parser.set_defaults(func=predict)

    walk_parser = subparsers.add_parser("walk_forward_train")
    walk_parser.add_argument("--panel-path", default="data/processed/panel_v2.parquet")
    walk_parser.add_argument("--features-path", default="data/processed/features_v2.parquet")
    walk_parser.add_argument("--config", default="config/automl_v2.yaml")
    walk_parser.set_defaults(func=walk_forward_train)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
