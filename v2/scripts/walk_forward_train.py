from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.automl_runner_v2 import _load_config, _prepare_features
from src.features_v2 import build_feature_matrix_v2


def walk_forward(args: argparse.Namespace) -> None:
    working_panel = pd.read_parquet(args.panel_path)
    config = _load_config(args.config)

    best_config_path = Path("models/best_config.json")
    if not best_config_path.exists():
        raise FileNotFoundError("best_config.json not found. Run run_xgb first.")
    best_config = json.loads(best_config_path.read_text(encoding="utf-8"))

    gpu_kwargs = config["fit_kwargs_by_estimator"]["xgboost"]

    forecast_months = sorted(working_panel.loc[working_panel["is_future"], "month"].unique())
    summaries = []

    for month in forecast_months:
        working_features, _ = build_feature_matrix_v2(working_panel, features_path=None, reports_dir=None)
        train_subset = working_features[working_features["month"] < month]
        if train_subset.empty:
            continue
        X_train, y_train_raw, _ = _prepare_features(train_subset)
        y_train_log = np.log1p(y_train_raw.to_numpy())

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        params = {**best_config, **gpu_kwargs}
        model = XGBRegressor(**params)
        model.fit(X_train_scaled, y_train_log)

        model_suffix = month.strftime("%Y-%m")
        Path("models").mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, Path("models") / f"scaler_{model_suffix}.pkl")
        joblib.dump(model, Path("models") / f"best_model_{model_suffix}.pkl")

        current_subset = working_features[working_features["month"] == month].copy()
        if current_subset.empty:
            continue
        X_iter = current_subset.drop(columns=["id", "sector", "month", "target", "target_filled", "is_future"], errors="ignore")
        preds_log = model.predict(scaler.transform(X_iter))
        preds = np.clip(np.expm1(preds_log), 0.0, None)

        working_panel.loc[working_panel["month"] == month, "target_filled"] = preds
        working_panel.loc[working_panel["month"] == month, "target"] = preds

        summaries.append({
            "month": month.strftime("%Y-%m"),
            "count": int(len(preds)),
            "mean": float(preds.mean()),
            "std": float(preds.std()),
            "min": float(preds.min()),
            "max": float(preds.max()),
        })

    Path("reports").mkdir(parents=True, exist_ok=True)
    with Path("reports/walk_forward_summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summaries, fp, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward retraining for v2")
    parser.add_argument("--panel-path", default="data/processed/panel_v2.parquet")
    parser.add_argument("--features-path", default="data/processed/features_v2.parquet")
    parser.add_argument("--config", default="config/automl_v2.yaml")
    args = parser.parse_args()
    walk_forward(args)


if __name__ == "__main__":
    main()
