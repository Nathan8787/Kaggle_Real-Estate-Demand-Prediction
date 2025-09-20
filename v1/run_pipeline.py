from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.automl_runner import run
from src.data_loading import load_raw_tables
from src.features import build_feature_matrix
from src.panel_builder import build_calendar, merge_sources, attach_target, save_panel
from src.predict import generate_submission
from src.splits import generate_time_series_folds


def build_panel_step(args: argparse.Namespace) -> None:
    config_path = Path(args.data_config)
    raw_tables = load_raw_tables(config_path)
    calendar_df = build_calendar(raw_tables)
    panel_df = merge_sources(calendar_df, raw_tables)
    panel_with_target = attach_target(panel_df)
    panel_path = Path(args.panel_path)
    save_panel(panel_with_target, panel_path)
    print(f"Panel saved to {panel_path}")


def build_features_step(args: argparse.Namespace) -> None:
    panel_path = Path(args.panel_path)
    if not panel_path.exists():
        raise FileNotFoundError('Panel parquet not found. Run build_panel first.')
    panel_df = pd.read_parquet(panel_path)
    build_feature_matrix(
        panel_df,
        features_path=Path(args.features_path),
        reports_dir=Path(args.reports_dir),
    )
    print(f"Features saved to {args.features_path}")


def run_automl_step(args: argparse.Namespace) -> None:
    features_path = Path(args.features_path)
    features_df = pd.read_parquet(features_path)
    train_df = features_df[features_df['month'] <= pd.Timestamp('2024-03-31')]
    folds = generate_time_series_folds(train_df)
    summary = run(features_df, folds, args.automl_config)
    print('AutoML summary:', summary)


def predict_step(args: argparse.Namespace) -> None:
    generate_submission(
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        features_path=args.features_path,
        sample_submission_path=args.sample_submission,
        output_path=args.output_path,
    )
    print(f"Submission written to {args.output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description='Pipeline runner for real estate demand prediction.')
    subparsers = parser.add_subparsers(dest='command', required=True)

    panel_parser = subparsers.add_parser('build_panel', help='Construct panel dataset.')
    panel_parser.add_argument('--data-config', default='config/data_paths.yaml')
    panel_parser.add_argument('--panel-path', default='data/processed/panel.parquet')
    panel_parser.set_defaults(func=build_panel_step)

    features_parser = subparsers.add_parser('build_features', help='Create feature matrix.')
    features_parser.add_argument('--panel-path', default='data/processed/panel.parquet')
    features_parser.add_argument('--features-path', default='data/processed/features.parquet')
    features_parser.add_argument('--reports-dir', default='reports')
    features_parser.set_defaults(func=build_features_step)

    automl_parser = subparsers.add_parser('run_automl', help='Execute AutoML search.')
    automl_parser.add_argument('--features-path', default='data/processed/features.parquet')
    automl_parser.add_argument('--automl-config', default='config/automl.yaml')
    automl_parser.set_defaults(func=run_automl_step)

    predict_parser = subparsers.add_parser('predict', help='Generate submission file.')
    predict_parser.add_argument('--model-path', default='models/best_model.pkl')
    predict_parser.add_argument('--scaler-path', default='models/scaler.pkl')
    predict_parser.add_argument('--features-path', default='data/processed/features.parquet')
    predict_parser.add_argument('--sample-submission', default='sample_submission.csv')
    predict_parser.add_argument('--output-path', default='submission/submission.csv')
    predict_parser.set_defaults(func=predict_step)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
