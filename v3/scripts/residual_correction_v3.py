from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

BASE_RESIDUAL_FEATURES = [
    "prediction",
    "month_index",
    "sector_id",
    "resident_population",
    "population_scale",
    "amount_new_house_transactions",
    "num_new_house_transactions",
    "amount_pre_owned_house_transactions",
    "num_pre_owned_house_transactions",
    "transaction_amount",
    "transaction_amount_nearby_sectors",
]


def _load_selected_search_features(report_path: Path) -> List[str]:
    if not report_path.exists():
        raise FileNotFoundError(f"Search keyword report not found: {report_path}")
    with report_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    selected = payload.get("selected") or payload.get("keywords")
    if not selected:
        raise ValueError("Search keyword report does not contain 'selected' entries")
    return [str(name) for name in selected]


def _load_predictions(predictions_dir: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted(predictions_dir.glob("predictions_fold_*.parquet")):
        frames.append(pd.read_parquet(path))
    holdout_path = predictions_dir / "predictions_holdout.parquet"
    if holdout_path.exists():
        frames.append(pd.read_parquet(holdout_path))
    if not frames:
        raise FileNotFoundError(
            f"No prediction files found in {predictions_dir}. Expected predictions_fold_*.parquet"
        )
    combined = pd.concat(frames, ignore_index=True)
    combined["month"] = pd.to_datetime(combined["month"], errors="coerce")
    return combined


def _prepare_training_frame(
    features_path: Path,
    predictions_dir: Path,
    search_report: Path,
) -> tuple[pd.DataFrame, List[str]]:
    predictions = _load_predictions(predictions_dir)
    features_df = pd.read_parquet(features_path)
    features_df["month"] = pd.to_datetime(features_df["month"], errors="coerce")

    merged = predictions.merge(
        features_df,
        on=["id", "month", "sector_id"],
        how="left",
        suffixes=("", "_feat"),
    )
    if "target" not in merged.columns:
        raise KeyError("Predictions must include the 'target' column")
    if "target_available_flag" not in merged.columns:
        flag_col = "target_available_flag_feat"
        if flag_col in merged.columns:
            merged["target_available_flag"] = merged[flag_col]
        else:
            raise KeyError("Missing target_available_flag for residual training")

    merged["residual"] = merged["target"].astype(float) - merged["prediction"].astype(float)
    training = merged[merged["target_available_flag"] == 1].copy()
    if training.empty:
        raise ValueError("No labeled rows available for residual training")

    search_features = _load_selected_search_features(search_report)
    residual_features = BASE_RESIDUAL_FEATURES + search_features

    missing = [col for col in residual_features if col not in training.columns]
    if missing:
        raise KeyError(f"Missing required residual features: {missing}")

    return training, residual_features


def _time_series_scores(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> list[dict[str, float]]:
    splitter = TimeSeriesSplit(n_splits=n_splits)
    metrics: list[dict[str, float]] = []

    for fold_id, (train_idx, valid_idx) in enumerate(splitter.split(X), start=1):
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)

        model = Ridge(alpha=0.5, random_state=42)
        model.fit(X_train_scaled, y_train)

        preds = model.predict(X_valid_scaled)
        residuals = y_valid - preds
        rmse = float(np.sqrt(np.mean(residuals**2)))
        mape = float(np.mean(np.abs(residuals) / (np.abs(y_valid) + 1e-6)))
        metrics.append({"fold": fold_id, "MAPE": mape, "RMSE": rmse})

    return metrics


def train_residual_model(args: argparse.Namespace) -> None:
    features_path = Path(args.features_path)
    predictions_dir = Path(args.predictions_dir)
    models_dir = Path(args.models_dir)
    reports_dir = Path(args.reports_dir)
    search_report = (
        Path(args.search_report)
        if args.search_report is not None
        else reports_dir / "search_keywords_v3.json"
    )

    training, residual_features = _prepare_training_frame(
        features_path, predictions_dir, search_report
    )
    training = training.sort_values("month").reset_index(drop=True)

    feature_matrix = training[residual_features].fillna(0.0).to_numpy(dtype=float)
    target_residuals = training["residual"].astype(float).to_numpy()

    if feature_matrix.shape[0] <= 5:
        raise ValueError("Residual training requires more than 5 observations for cross-validation")

    fold_metrics = _time_series_scores(feature_matrix, target_residuals)
    avg_mape = float(np.mean([entry["MAPE"] for entry in fold_metrics])) if fold_metrics else float("nan")

    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "status": "accepted" if avg_mape < 0.40 else "rejected",
        "average_mape": avg_mape,
        "folds": fold_metrics,
    }

    metrics_path = reports_dir / "residual_training_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2, ensure_ascii=False)

    if metrics_payload["status"] == "rejected":
        return

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)
    model = Ridge(alpha=0.5, random_state=42)
    model.fit(scaled_features, target_residuals)

    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, models_dir / "residual_scaler.pkl")
    joblib.dump(model, models_dir / "residual_model.pkl")

    feature_list = sorted(residual_features)
    with (reports_dir / "residual_features.json").open("w", encoding="utf-8") as handle:
        json.dump(feature_list, handle, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train residual correction model for v3 pipeline")
    parser.add_argument("--features-path", required=True, help="Path to features parquet file")
    parser.add_argument(
        "--predictions-dir",
        required=True,
        help="Directory containing fold and holdout prediction parquet files",
    )
    parser.add_argument("--models-dir", default="models_v3", help="Directory to store residual artifacts")
    parser.add_argument("--reports-dir", default="reports_v3", help="Directory for residual reports")
    parser.add_argument(
        "--search-report",
        default=None,
        help="Optional path to search keyword selection report; defaults to reports_dir/search_keywords_v3.json",
    )
    args = parser.parse_args()
    train_residual_model(args)


if __name__ == "__main__":
    main()
