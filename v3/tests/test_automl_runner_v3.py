from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.automl_runner_v3 import run_cross_validation
from src import splits_v3


def _load_feature_subset() -> pd.DataFrame:
    root = ROOT.parent
    transactions_path = root / "train" / "new_house_transactions.csv"
    sector_map_path = root / "v3" / "config" / "sector_city_map_v3.csv"

    df = pd.read_csv(transactions_path, encoding="utf-8-sig")
    df["month"] = pd.to_datetime(df["month"], format="%Y-%b")
    df["sector_id"] = df["sector"].str.extract(r"(\d+)").astype(int)
    df = df[df["sector_id"].isin([1])].copy()

    sector_map = pd.read_csv(sector_map_path, encoding="utf-8-sig")
    df = df.merge(sector_map, on="sector", how="left", validate="many_to_one")

    df["target"] = df["amount_new_house_transactions"].astype(float)
    df["target_filled"] = df["target"]
    df["target_available_flag"] = (
        df["month"] <= pd.Timestamp("2024-07-31")
    ).astype("int8")
    df["target_filled_was_missing"] = 0
    df["is_future"] = (df["month"] > pd.Timestamp("2024-07-31")).astype("int8")
    df["city_id"] = df["city_id"].astype("int16")
    df["sector_id"] = df["sector_id"].astype("int32")

    df = df[df["month"] <= pd.Timestamp("2024-07-01")]
    df = df.drop(columns=["sector"])
    return df


def test_run_cross_validation_smoke(monkeypatch, tmp_path: Path):
    features_df = _load_feature_subset()

    monkeypatch.setattr(splits_v3, "FOLD_DEFINITIONS_V3", [splits_v3.FOLD_DEFINITIONS_V3[0]])

    from src import automl_runner_v3

    class DummyAutoML:
        def __init__(self):
            self.best_config = {
                "n_estimators": 20,
                "max_depth": 3,
                "learning_rate": 0.1,
            }
            self.best_iteration = 1
            self.config_history = {0: ("xgboost", self.best_config, 0.1)}

        def add_metric(self, name, func, greater_is_better=True):
            return None

        def fit(self, **kwargs):
            self.last_fit_params = kwargs
            return self

    monkeypatch.setattr(automl_runner_v3, "AutoML", DummyAutoML)

    config_path = tmp_path / "automl_test.yaml"
    config_path.write_text(
        """
seed: 42
time_budget_per_fold: 5
min_trials_per_fold: 1
metric: competition_score
task: regression
estimator_list: ["xgboost"]
n_jobs: 1
gpu_per_trial: null
fit_kwargs_by_estimator:
  xgboost:
    tree_method: hist
    objective: reg:tweedie
    tweedie_variance_power: 1.3
search_space:
  xgboost:
    n_estimators:
      _type: randint
      _value: [20, 40]
    max_depth:
      _type: randint
      _value: [3, 4]
    learning_rate:
      _type: uniform
      _value: [0.05, 0.1]
    min_child_weight:
      _type: uniform
      _value: [1.0, 3.0]
    subsample:
      _type: uniform
      _value: [0.7, 1.0]
    colsample_bytree:
      _type: uniform
      _value: [0.6, 1.0]
    reg_alpha:
      _type: uniform
      _value: [0.0, 0.1]
    reg_lambda:
      _type: uniform
      _value: [0.5, 1.5]
zero_inflation:
  enabled: false
  method: tweedie
""",
        encoding="utf-8",
    )

    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    logs_dir = tmp_path / "logs"

    fold_results, summary = run_cross_validation(
        features_df,
        config_path=config_path,
        models_dir=models_dir,
        reports_dir=reports_dir,
        logs_dir=logs_dir,
    )

    assert fold_results
    assert summary["xgboost"]["fold_scores"], "Cross-validation should record fold scores"
    assert (reports_dir / "fold_metrics_v3.csv").exists()
