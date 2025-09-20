from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from time import perf_counter
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None
from flaml import AutoML, tune
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .metrics_v2 import competition_metric, competition_score
from .splits_v2 import generate_time_series_folds_v2

__all__ = ["run", "SingleFoldSplitter", "FoldResult"]


@dataclass
class FoldResult:
    fold: int
    best_estimator: str
    best_iteration: int | None
    validation_score: float


class SingleFoldSplitter:
    def __init__(self, train_idx: np.ndarray, valid_idx: np.ndarray):
        self.train_idx = np.asarray(train_idx, dtype=int)
        self.valid_idx = np.asarray(valid_idx, dtype=int)

    def split(self, X=None, y=None, groups=None):
        yield self.train_idx, self.valid_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1

def _load_config(config_path: str | Path) -> Dict:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def _convert_search_space(search_space: Dict | None) -> Dict | None:
    if not search_space:
        return None

    converted = {}
    for estimator, params in search_space.items():
        new_params: Dict[str, Dict[str, object]] = {}
        for name, spec in params.items():
            spec_type = spec.get("_type") if isinstance(spec, dict) else None
            values = spec.get("_value") if isinstance(spec, dict) else None
            if spec_type == "randint" and isinstance(values, (list, tuple)) and len(values) == 2:
                low, high = int(values[0]), int(values[1])
                domain = tune.randint(lower=low, upper=high)
                new_params[name] = {"domain": domain, "init_value": low}
            elif spec_type == "uniform" and isinstance(values, (list, tuple)) and len(values) == 2:
                low, high = float(values[0]), float(values[1])
                domain = tune.uniform(lower=low, upper=high)
                new_params[name] = {"domain": domain, "init_value": (low + high) / 2}
            elif spec_type == "loguniform" and isinstance(values, (list, tuple)) and len(values) == 2:
                low, high = float(values[0]), float(values[1])
                domain = tune.loguniform(lower=low, upper=high)
                init = float(np.sqrt(low * high)) if low > 0 and high > 0 else low
                new_params[name] = {"domain": domain, "init_value": init}
            else:
                new_params[name] = spec
        converted[estimator] = new_params
    return converted




def _prepare_features(features_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    exclude = {"id", "sector", "month", "target", "target_filled", "is_future"}
    feature_cols = [col for col in features_df.columns if col not in exclude]
    X = features_df[feature_cols].astype(float)
    y = features_df["target"].astype(float)
    return X, y, feature_cols


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def run(features_df: pd.DataFrame, config_path: str | Path) -> Dict[str, List[str]]:
    config = _load_config(config_path)

    fit_kwargs = config.get("fit_kwargs_by_estimator", {}).get("xgboost", {})
    constant_params = {}
    for key in list(fit_kwargs.keys()):
        if key in {"tree_method", "predictor", "gpu_id", "device"}:
            constant_params[key] = fit_kwargs.pop(key)

    search_space = _convert_search_space(config.get("search_space"))
    if constant_params:
        if search_space is None:
            search_space = {}
        search_space.setdefault("xgboost", {})
        for key, value in constant_params.items():
            search_space["xgboost"][key] = {"domain": tune.choice([value]), "init_value": value}


    logs_path = Path("logs/automl.log")
    logs_path.parent.mkdir(parents=True, exist_ok=True)

    train_cutoff = pd.Timestamp("2024-07-01")
    train_df = features_df[features_df["month"] <= train_cutoff].copy()

    X_full, y_full_raw, feature_cols = _prepare_features(train_df)
    y_full_log = np.log1p(y_full_raw.to_numpy())

    folds = generate_time_series_folds_v2(train_df)
    total_folds = len(folds) if folds else 1
    time_budget = float(config.get("time_budget", 0.0) or 0.0)
    max_iter = config.get("max_iter")

    fold_time_budget = time_budget / total_folds if time_budget > 0 else None
    if fold_time_budget:
        print(f"[AutoML] Allocated {fold_time_budget:.1f}s per fold across {total_folds} folds.")
    elif max_iter:
        print(f"[AutoML] Using max_iter={max_iter} for each of {total_folds} folds.")
    else:
        print(f"[AutoML] Using AutoML default settings for {total_folds} folds.")

    progress = tqdm(total=total_folds, desc="AutoML folds", unit="fold") if tqdm else None

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    fold_predictions: List[pd.DataFrame] = []
    fold_summaries: List[FoldResult] = []
    automl_objects: List[AutoML] = []

    for fold_id, (train_idx, valid_idx) in enumerate(folds, start=1):
        if progress:
            progress.set_postfix_str(f"fold {fold_id}/{total_folds} starting" + (f" (budget {fold_time_budget:.1f}s)" if fold_time_budget else ""))
        else:
            print(f"[AutoML] Starting fold {fold_id}/{total_folds}" + (f" (budget {fold_time_budget:.1f}s)" if fold_time_budget else ""))
        fold_start = perf_counter()

        fold_subset = train_df.loc[list(train_idx) + list(valid_idx)].sort_values(["month", "sector_id"]).reset_index()
        index_map = fold_subset["index"].to_numpy()

        fold_X = X_full.loc[index_map, :].reset_index(drop=True)
        fold_y = y_full_raw.loc[index_map].reset_index(drop=True)
        fold_y_log = np.log1p(fold_y.to_numpy())

        train_mask = np.isin(index_map, train_idx)
        valid_mask = np.isin(index_map, valid_idx)
        train_positions = np.where(train_mask)[0]
        valid_positions = np.where(valid_mask)[0]

        scaler = StandardScaler()
        scaler.fit(fold_X.iloc[train_positions])
        fold_X_scaled = pd.DataFrame(scaler.transform(fold_X), columns=feature_cols)

        _write_json(reports_dir / f"scaler_fold_{fold_id}.json", {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        })

        splitter = SingleFoldSplitter(train_positions, valid_positions)
        automl = AutoML()
        automl_settings: Dict[str, object] = {
            "metric": competition_metric,
            "task": "regression",
            "estimator_list": config["estimator_list"],
            "log_file_name": str(logs_path),
            "seed": config["seed"],
            "n_jobs": config["n_jobs"],
            "eval_method": "cv",
            "split_type": splitter,
            "fit_kwargs_by_estimator": config.get("fit_kwargs_by_estimator"),
            "custom_hp": search_space,
            "verbose": 1,
        }
        if fold_time_budget:
            automl_settings["time_budget"] = fold_time_budget
        elif time_budget > 0:
            automl_settings["time_budget"] = time_budget
        if max_iter:
            automl_settings["max_iter"] = int(max_iter)

        automl.fit(X_train=fold_X_scaled, y_train=fold_y_log, **automl_settings)

        if automl.best_estimator is None:
            if progress:
                progress.close()
            raise RuntimeError(
                "FLAML did not train any estimator for fold "                + f"{fold_id}. Increase time_budget or relax the search space."
            )

        y_valid_pred_log = automl.predict(fold_X_scaled.iloc[valid_positions])
        y_valid_pred = np.clip(np.expm1(y_valid_pred_log), 0.0, None)
        y_valid_true = fold_y.iloc[valid_positions].to_numpy()
        score = competition_score(y_valid_true, y_valid_pred)

        elapsed = perf_counter() - fold_start
        if progress:
            progress.update(1)
            progress.set_postfix_str(f"fold {fold_id}/{total_folds} score {score:.4f} ({elapsed:.1f}s)")
        else:
            print(f"[AutoML] Completed fold {fold_id}/{total_folds} in {elapsed:.1f}s with score {score:.4f}")

        fold_predictions.append(pd.DataFrame({
            "fold": fold_id,
            "id": train_df.loc[valid_idx, "id"].values,
            "month": train_df.loc[valid_idx, "month"].values,
            "sector_id": train_df.loc[valid_idx, "sector_id"].values,
            "target": y_valid_true,
            "prediction": y_valid_pred,
            "fold_score": score,
        }))

        fold_summaries.append(
            FoldResult(
                fold=fold_id,
                best_estimator=automl.best_estimator,
                best_iteration=automl.best_iteration,
                validation_score=float(score),
            )
        )
        automl_objects.append(automl)

    fold_results = pd.concat(fold_predictions, axis=0, ignore_index=True)
    fold_results.to_csv(reports_dir / "fold_results.csv", index=False)

    best_idx = int(np.argmax([summary.validation_score for summary in fold_summaries]))
    best_automl = automl_objects[best_idx]
    best_config = best_automl.best_config.copy()
    _write_json(Path("models") / "best_config.json", best_config)

    global_scaler = StandardScaler()
    global_scaler.fit(X_full)
    X_full_scaled = global_scaler.transform(X_full)

    Path("models").mkdir(parents=True, exist_ok=True)
    joblib.dump(global_scaler, Path("models") / "scaler.pkl")

    if progress:
        progress.close()
    print("[AutoML] Training final model on full dataset...")

    base_fit_kwargs = dict(config.get("fit_kwargs_by_estimator", {}).get("xgboost", {}))
    final_params = dict(best_config)
    final_params.update(base_fit_kwargs)
    final_params.update(constant_params)
    final_params.pop("predictor", None)
    final_params.setdefault("objective", "reg:squarederror")

    final_model = XGBRegressor(**final_params)
    final_model.fit(X_full_scaled, y_full_log)
    print("[AutoML] Final model training complete.")
    joblib.dump(final_model, Path("models") / "best_model.pkl")

    summary_payload = {
        "folds": [summary.__dict__ for summary in fold_summaries],
        "best_estimator": fold_summaries[best_idx].best_estimator,
        "best_fold": fold_summaries[best_idx].fold,
        "feature_columns": feature_cols,
        "removed_estimators": [],
        "target_transform": "log1p_expm1",
        "best_config": best_config,
    }
    _write_json(reports_dir / "automl_summary.json", summary_payload)

    log_entry = {
        "best_estimator": fold_summaries[best_idx].best_estimator,
        "fold_scores": {summary.fold: summary.validation_score for summary in fold_summaries},
    }
    with logs_path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(log_entry) + "\n")

    return summary_payload
