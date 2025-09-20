from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

DEFAULT_FEATURES = [
    "prediction",
    "month_index",
    "target_city_weighted_mean",
    "target_share",
]


def train_residual_model(args: argparse.Namespace) -> None:
    fold_results_path = Path(args.fold_results)
    features_path = Path(args.features_path)

    if not fold_results_path.exists():
        raise FileNotFoundError(f"Fold results not found: {fold_results_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    fold_results = pd.read_csv(fold_results_path)
    features_df = pd.read_parquet(features_path)

    merged = fold_results.merge(features_df, on=["id", "month", "sector_id"], how="left", suffixes=("", "_feat"))
    merged["residual"] = merged["target"] - merged["prediction"]

    missing_features = [col for col in DEFAULT_FEATURES if col not in merged.columns]
    if missing_features:
        raise KeyError(f"Missing required residual features: {missing_features}")

    X = merged[DEFAULT_FEATURES].fillna(0.0)
    y = merged["residual"].fillna(0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = Ridge(alpha=1.0)
    model.fit(X_scaled, y)

    Path(args.output_model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output_model)
    joblib.dump(scaler, args.output_scaler)

    Path(args.output_features).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output_features).open("w", encoding="utf-8") as fp:
        json.dump({"features": DEFAULT_FEATURES}, fp, indent=2)

    print(f"Residual model saved to {args.output_model}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train residual correction model")
    parser.add_argument("--fold-results", default="reports/fold_results.csv")
    parser.add_argument("--features-path", default="data/processed/features_v2.parquet")
    parser.add_argument("--output-model", default="models/residual_model.pkl")
    parser.add_argument("--output-scaler", default="models/residual_scaler.pkl")
    parser.add_argument("--output-features", default="reports/residual_features.json")
    args = parser.parse_args()
    train_residual_model(args)


if __name__ == "__main__":
    main()
