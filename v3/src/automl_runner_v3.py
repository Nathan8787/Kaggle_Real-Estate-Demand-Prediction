from __future__ import annotations

import json
import logging
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from flaml import AutoML
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

from .metrics_v3 import competition_metric, competition_score
from .splits_v3 import HOLDOUT_WINDOW, generate_time_series_folds_v3

logger = logging.getLogger(__name__)

__all__ = [
    "run_cross_validation",
    "train_holdout_model",
    "train_full_model",
    "run_training_pipeline",
]

RECENT_WEIGHT_START = pd.Timestamp("2024-04-01")
RECENT_WEIGHT_END = pd.Timestamp("2024-06-30")
BASE_SAMPLE_WEIGHT = 1.0
RECENT_SAMPLE_WEIGHT = 1.8

MONOTONE_POSITIVE_COLUMNS = {
    "resident_population",
    "population_scale",
    "surrounding_housing_average_price",
    "surrounding_shop_average_rent",
    "city_gdp_100m",
    "city_total_retail_sales_of_consumer_goods_100m",
}

ESTIMATOR_ALIAS = {"xgboost": "xgb", "lgbm": "lgbm"}


@dataclass
class FoldModelResult:
    fold_id: int
    estimator: str
    config: Dict[str, object]
    best_iteration: Optional[int]
    metrics: Dict[str, float]
    model_path: Path
    importance_path: Path


def _load_config(config_path: Path) -> Dict[str, object]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    required_keys = {
        "seed",
        "time_budget_per_fold",
        "min_trials_per_fold",
        "metric",
        "task",
        "estimator_list",
        "n_jobs",
        "gpu_per_trial",
        "fit_kwargs_by_estimator",
        "search_space",
    }
    missing = required_keys - set(config)
    if missing:
        raise KeyError(f"Missing required keys in automl config: {sorted(missing)}")

    fit_kwargs = config.get("fit_kwargs_by_estimator", {})
    search_space = config.get("search_space", {})
    for estimator in config["estimator_list"]:
        if estimator not in fit_kwargs:
            raise KeyError(f"fit_kwargs_by_estimator missing entry for '{estimator}'")
        if estimator not in search_space:
            raise KeyError(f"search_space missing entry for '{estimator}'")
    return config


def _ensure_directories(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def _select_feature_columns(features_df: pd.DataFrame) -> List[str]:
    exclude = {"month", "id", "target"}
    numeric_cols = [
        col
        for col in features_df.columns
        if col not in exclude and pd.api.types.is_numeric_dtype(features_df[col])
    ]
    if not numeric_cols:
        raise ValueError("No numeric feature columns available for training")
    return numeric_cols


def _compute_sample_weight(month_series: pd.Series) -> np.ndarray:
    months = pd.to_datetime(month_series)
    months = months.dt.to_period("M").dt.to_timestamp()
    mask_recent = months.between(RECENT_WEIGHT_START, RECENT_WEIGHT_END, inclusive="both")
    weights = np.full(months.shape, BASE_SAMPLE_WEIGHT, dtype=float)
    weights[mask_recent.to_numpy()] = RECENT_SAMPLE_WEIGHT
    return weights


def _build_monotone_constraints(feature_cols: Sequence[str]) -> List[int]:
    constraints = [1 if col in MONOTONE_POSITIVE_COLUMNS else 0 for col in feature_cols]
    return constraints


def _format_xgb_monotone(constraints: Sequence[int]) -> str:
    return "(" + ",".join(str(int(value)) for value in constraints) + ")"


def _register_metric(automl: AutoML) -> str:
    if hasattr(automl, "add_metric"):
        try:
            automl.add_metric("competition_score", competition_metric, greater_is_better=True)
            return "competition_score"
        except TypeError:
            automl.add_metric("competition_score", competition_metric)
            return "competition_score"
    logger.warning("AutoML.add_metric unavailable; passing metric callable directly")
    return competition_metric


def _log_trials(automl: AutoML, fold_id: int, estimator: str, logs_path: Path) -> None:
    rows: List[Dict[str, object]] = []
    for iteration, payload in automl.config_history.items():
        if not payload:
            continue
        trial_estimator = payload[0]
        config = payload[1] if len(payload) > 1 else {}
        wall_time = payload[2] if len(payload) > 2 else None
        rows.append(
            {
                "fold": fold_id,
                "estimator": estimator,
                "suggested_estimator": trial_estimator,
                "iteration": iteration,
                "wall_time": float(wall_time) if wall_time is not None else None,
                "config": json.dumps(config),
            }
        )
    if not rows:
        return
    logs_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if logs_path.exists():
        existing = pd.read_csv(logs_path)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(logs_path, index=False)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0.0, None)
    error = y_pred - y_true
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(np.mean(error**2)))
    mape = float(np.mean(np.abs(error) / (np.maximum(np.abs(y_true), 1e-6))))
    smape = float(
        np.mean(
            2.0
            * np.abs(error)
            / (np.abs(y_true) + np.abs(y_pred) + 1e-6)
        )
    )
    score = competition_score(y_true, y_pred)
    r2 = r2_score(y_true, y_pred) if y_true.size > 1 else float("nan")
    return {
        "competition_score": float(score),
        "SMAPE": smape,
        "MAPE": mape,
        "MAE": float(mae),
        "RMSE": rmse,
        "R2": float(r2),
    }


def _prepare_params(
    estimator: str,
    base_config: Dict[str, object],
    fit_kwargs: Dict[str, object],
    seed: int,
) -> Dict[str, object]:
    params = {**base_config}
    params.update(fit_kwargs)
    params.setdefault("random_state", seed)
    params.setdefault("n_estimators", base_config.get("n_estimators", 500))
    if estimator == "xgboost":
        params.setdefault("objective", "reg:tweedie")
        params.setdefault("tweedie_variance_power", 1.3)
    elif estimator == "lgbm":
        params.setdefault("objective", "tweedie")
        params.setdefault("tweedie_variance_power", 1.3)
    return params


def _train_estimator(
    estimator: str,
    params: Dict[str, object],
    monotone_constraints: Sequence[int],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    sample_weight_train: np.ndarray,
    X_valid: Optional[pd.DataFrame] = None,
    y_valid: Optional[np.ndarray] = None,
    sample_weight_valid: Optional[np.ndarray] = None,
) -> Tuple[object, Optional[np.ndarray]]:
    if estimator == "xgboost":
        params = params.copy()
        params["monotone_constraints"] = _format_xgb_monotone(monotone_constraints)
        model = XGBRegressor(**params)
        fit_kwargs = {"sample_weight": sample_weight_train}
        if X_valid is not None and y_valid is not None:
            fit_kwargs["eval_set"] = [(X_valid, y_valid)]
            if sample_weight_valid is not None:
                fit_kwargs["sample_weight_eval_set"] = [sample_weight_valid]
            fit_kwargs["early_stopping_rounds"] = 100
            fit_kwargs["verbose"] = False
        try:
            model.fit(X_train, y_train, **fit_kwargs)
        except TypeError:
            fit_kwargs.pop("early_stopping_rounds", None)
            fit_kwargs.pop("verbose", None)
            fit_kwargs.pop("sample_weight_eval_set", None)
            model.fit(X_train, y_train, **fit_kwargs)
        predictions = model.predict(X_valid) if X_valid is not None else None
    elif estimator == "lgbm":
        params = params.copy()
        params["monotone_constraints"] = list(monotone_constraints)
        model = LGBMRegressor(**params)
        fit_kwargs = {"sample_weight": sample_weight_train}
        if X_valid is not None and y_valid is not None:
            fit_kwargs["eval_set"] = [(X_valid, y_valid)]
            if sample_weight_valid is not None:
                fit_kwargs["eval_sample_weight"] = [sample_weight_valid]
            fit_kwargs["verbose"] = -1
            fit_kwargs["callbacks"] = []
        try:
            model.fit(X_train, y_train, **fit_kwargs)
        except TypeError:
            fit_kwargs.pop("callbacks", None)
            model.fit(X_train, y_train, **fit_kwargs)
        predictions = model.predict(X_valid) if X_valid is not None else None
    else:
        raise ValueError(f"Unsupported estimator: {estimator}")
    return model, predictions


def _save_feature_importance(
    estimator: str,
    model,
    feature_cols: Sequence[str],
    output_path: Path,
) -> pd.DataFrame:
    if estimator == "xgboost":
        booster = model.get_booster()
        importance_map = booster.get_score(importance_type="gain")
        data = sorted(importance_map.items(), key=lambda item: item[1], reverse=True)
    else:
        booster = model.booster_
        gains = booster.feature_importance(importance_type="gain")
        names = booster.feature_name()
        data = sorted(zip(names, gains), key=lambda item: item[1], reverse=True)
    df = pd.DataFrame(data, columns=["feature", "importance"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def _save_model(estimator: str, model, model_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if estimator == "xgboost":
        model.save_model(model_path)
    else:
        model.booster_.save_model(str(model_path))


def _get_model_path(models_dir: Path, fold_id: int, estimator: str, phase: str) -> Path:
    alias = ESTIMATOR_ALIAS.get(estimator, estimator)
    suffix = "json" if estimator == "xgboost" else "txt"
    return models_dir / f"{phase}_{fold_id}_{alias}.{suffix}"


def _prepare_predictions_frame(
    base_df: pd.DataFrame,
    predictions: np.ndarray,
    sample_weight: np.ndarray,
    model_type: str,
) -> pd.DataFrame:
    ids = base_df.get("id")
    if ids is None:
        ids = (
            base_df["month"].dt.strftime("%Y-%m")
            + "_sector "
            + base_df["sector_id"].astype(str)
        )
    frame = pd.DataFrame(
        {
            "id": ids,
            "month": base_df["month"],
            "sector_id": base_df["sector_id"],
            "target": base_df["target"],
            "prediction": np.clip(predictions, 0.0, None),
            "sample_weight": sample_weight,
            "model_type": model_type,
        }
    )
    return frame


def run_cross_validation(
    features_df: pd.DataFrame,
    config_path: Path | str,
    models_dir: Path | str,
    reports_dir: Path | str,
    logs_dir: Path | str,
) -> Tuple[List[FoldModelResult], Dict[str, object]]:
    config = _load_config(Path(config_path))
    models_dir = Path(models_dir)
    reports_dir = Path(reports_dir)
    logs_dir = Path(logs_dir)
    _ensure_directories(models_dir, reports_dir, logs_dir)

    ordered_df = features_df.copy()
    ordered_df["month"] = pd.to_datetime(ordered_df["month"], errors="coerce")
    ordered_df["month"] = ordered_df["month"].dt.to_period("M").dt.to_timestamp()

    folds = generate_time_series_folds_v3(ordered_df, reports_dir=reports_dir)
    feature_cols = _select_feature_columns(ordered_df)
    monotone_vector = _build_monotone_constraints(feature_cols)

    fold_results: List[FoldModelResult] = []
    fold_metrics_rows: List[Dict[str, object]] = []
    estimator_summary: Dict[str, Dict[str, object]] = {
        estimator: {"fold_scores": [], "best_config": None, "best_iteration": None, "best_score": -np.inf}
        for estimator in config["estimator_list"]
    }
    ensemble_scores: List[float] = []

    for fold_id, (train_idx, valid_idx) in enumerate(folds, start=1):
        train_slice = ordered_df.loc[train_idx].copy()
        valid_slice = ordered_df.loc[valid_idx].copy()

        train_slice = train_slice[train_slice["target"].notna()]
        valid_slice = valid_slice[valid_slice["target"].notna()]
        if train_slice.empty or valid_slice.empty:
            logger.warning("Fold %s skipped due to insufficient labelled data", fold_id)
            continue

        X_train = train_slice[feature_cols]
        X_valid = valid_slice[feature_cols]
        y_train = train_slice["target"].astype(float).to_numpy()
        y_valid = valid_slice["target"].astype(float).to_numpy()

        sample_weight_train = _compute_sample_weight(train_slice["month"])
        sample_weight_valid = _compute_sample_weight(valid_slice["month"])

        fold_weight_stats = {
            "avg_sample_weight": float(sample_weight_train.mean()),
            "weighted_rows": float(sample_weight_train.sum()),
        }

        predictions_by_estimator: Dict[str, np.ndarray] = {}

        for estimator in config["estimator_list"]:
            automl = AutoML()
            metric_name = _register_metric(automl)
            base_params = copy.deepcopy(
                config["fit_kwargs_by_estimator"].get(estimator, {})
            )
            estimator_space = copy.deepcopy(
                config["search_space"].get(estimator, {})
            )

            def _set_constant(space: dict, key: str, value) -> None:
                space[key] = {"_type": "choice", "_value": [value]}

            for key, value in base_params.items():
                _set_constant(estimator_space, key, value)
            if estimator == "xgboost":
                _set_constant(
                    estimator_space, "monotone_constraints", _format_xgb_monotone(monotone_vector)
                )
            else:
                _set_constant(
                    estimator_space, "monotone_constraints", list(monotone_vector)
                )

            fit_kwargs = {estimator: {}}
            search_space = {estimator: estimator_space}
            fit_params = {
                "X_train": X_train,
                "y_train": y_train,
                "task": config["task"],
                "metric": metric_name,
                "estimator_list": [estimator],
                "time_budget": float(config["time_budget_per_fold"]),
                "n_jobs": int(config["n_jobs"]),
                "sample_weight": sample_weight_train,
                "eval_method": "holdout",
                "X_val": X_valid,
                "y_val": y_valid,
                "seed": int(config["seed"]),
                "verbose": 0,
                "fit_kwargs_by_estimator": fit_kwargs,
                "search_space": search_space,
            }
            if config.get("gpu_per_trial") is not None:
                fit_params["gpu_per_trial"] = config.get("gpu_per_trial")
            automl.fit(**fit_params)
            best_iteration = getattr(automl, "best_iteration", None)
            if best_iteration is not None and best_iteration < int(config["min_trials_per_fold"]):
                extra_budget = max(0.0, 1200.0 - float(config["time_budget_per_fold"]))
                if extra_budget > 0:
                    automl.fit(
                        X_train=X_train,
                        y_train=y_train,
                        task=config["task"],
                        metric=metric_name,
                    estimator_list=[estimator],
                    time_budget=extra_budget,
                    n_jobs=int(config["n_jobs"]),
                    sample_weight=sample_weight_train,
                    eval_method="holdout",
                    X_val=X_valid,
                    y_val=y_valid,
                    seed=int(config["seed"]),
                    verbose=0,
                    fit_kwargs_by_estimator=fit_kwargs,
                    search_space=search_space,
                )
                    best_iteration = getattr(automl, "best_iteration", best_iteration)

            best_config = automl.best_config.copy()
            params = _prepare_params(
                estimator,
                best_config,
                config["fit_kwargs_by_estimator"].get(estimator, {}),
                int(config["seed"]),
            )
            model, predictions = _train_estimator(
                estimator,
                params,
                monotone_vector,
                X_train,
                y_train,
                sample_weight_train,
                X_valid,
                y_valid,
                sample_weight_valid,
            )
            if predictions is None:
                raise RuntimeError("Validation predictions missing after model fit")
            predictions = np.clip(predictions, 0.0, None)
            alias = ESTIMATOR_ALIAS.get(estimator, estimator)
            model_path = _get_model_path(models_dir, fold_id, estimator, "fold")
            _save_model(estimator, model, model_path)
            importance_path = reports_dir / f"feature_importance_fold_{fold_id}_{alias}.csv"
            _save_feature_importance(estimator, model, feature_cols, importance_path)

            metrics = _compute_metrics(y_valid, predictions)
            estimator_summary_entry = estimator_summary[estimator]
            estimator_summary_entry["fold_scores"].append(metrics["competition_score"])
            if metrics["competition_score"] > estimator_summary_entry["best_score"]:
                estimator_summary_entry["best_score"] = metrics["competition_score"]
                estimator_summary_entry["best_config"] = best_config
                estimator_summary_entry["best_iteration"] = best_iteration

            fold_results.append(
                FoldModelResult(
                    fold_id=fold_id,
                    estimator=estimator,
                    config=best_config,
                    best_iteration=best_iteration,
                    metrics=metrics,
                    model_path=model_path,
                    importance_path=importance_path,
                )
            )

            prediction_frame = _prepare_predictions_frame(
                valid_slice, predictions, sample_weight_valid, alias
            )
            prediction_path = reports_dir / f"predictions_fold_{fold_id}_{alias}.parquet"
            prediction_frame.to_parquet(prediction_path, index=False)

            fold_metrics_rows.append(
                {
                    "fold": fold_id,
                    "model_type": alias,
                    **metrics,
                    **fold_weight_stats,
                }
            )
            predictions_by_estimator[alias] = predictions

            _log_trials(automl, fold_id, estimator, logs_dir / "automl_trials.csv")

        if predictions_by_estimator:
            ensemble_pred = np.mean(list(predictions_by_estimator.values()), axis=0)
            ensemble_metrics = _compute_metrics(y_valid, ensemble_pred)
            ensemble_scores.append(ensemble_metrics["competition_score"])
            ensemble_frame = _prepare_predictions_frame(
                valid_slice, ensemble_pred, sample_weight_valid, "ensemble"
            )
            ensemble_path = reports_dir / f"predictions_fold_{fold_id}_ensemble.parquet"
            ensemble_frame.to_parquet(ensemble_path, index=False)
            fold_metrics_rows.append(
                {
                    "fold": fold_id,
                    "model_type": "ensemble",
                    **ensemble_metrics,
                    **fold_weight_stats,
                }
            )

    if fold_metrics_rows:
        metrics_df = pd.DataFrame(fold_metrics_rows)
        metrics_df.to_csv(reports_dir / "fold_metrics_v3.csv", index=False)

    summary = {
        estimator: {
            "best_config": estimator_summary[estimator]["best_config"] or {},
            "best_iteration": estimator_summary[estimator]["best_iteration"],
            "fold_scores": estimator_summary[estimator]["fold_scores"],
            "avg_score": float(np.mean(estimator_summary[estimator]["fold_scores"]))
            if estimator_summary[estimator]["fold_scores"]
            else None,
            "monotone_constraints": list(monotone_vector),
        }
        for estimator in config["estimator_list"]
    }
    summary.update(
        {
            "ensemble_score": float(np.mean(ensemble_scores)) if ensemble_scores else None,
            "feature_count": len(feature_cols),
            "sample_weight_strategy": {
                "base_weight": BASE_SAMPLE_WEIGHT,
                "recent_weight": RECENT_SAMPLE_WEIGHT,
                "recent_window": [
                    RECENT_WEIGHT_START.strftime("%Y-%m-%d"),
                    RECENT_WEIGHT_END.strftime("%Y-%m-%d"),
                ],
            },
        }
    )
    with (reports_dir / "automl_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    return fold_results, summary


def train_holdout_model(
    features_df: pd.DataFrame,
    best_configs: Dict[str, Dict[str, object]],
    models_dir: Path | str,
    reports_dir: Path | str,
    config: Dict[str, object],
) -> Dict[str, object]:
    models_dir = Path(models_dir)
    reports_dir = Path(reports_dir)
    _ensure_directories(models_dir, reports_dir)

    df = features_df.copy()
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df["month"] = df["month"].dt.to_period("M").dt.to_timestamp()

    train_start, train_end, holdout_start, holdout_end = [pd.Timestamp(v) for v in HOLDOUT_WINDOW]
    train_mask = (df["month"] >= train_start) & (df["month"] <= train_end) & df["target"].notna()
    valid_mask = (df["month"] >= holdout_start) & (df["month"] <= holdout_end) & df["target"].notna()

    train_slice = df.loc[train_mask].copy()
    valid_slice = df.loc[valid_mask].copy()
    if train_slice.empty or valid_slice.empty:
        raise ValueError("Holdout training requires labelled data through 2024-07")

    feature_cols = _select_feature_columns(df)
    monotone_vector = _build_monotone_constraints(feature_cols)

    X_train = train_slice[feature_cols]
    y_train = train_slice["target"].astype(float).to_numpy()
    X_valid = valid_slice[feature_cols]
    y_valid = valid_slice["target"].astype(float).to_numpy()

    sample_weight_train = _compute_sample_weight(train_slice["month"])
    sample_weight_valid = _compute_sample_weight(valid_slice["month"])

    metrics_payload: Dict[str, Dict[str, float]] = {}
    predictions_payload: Dict[str, np.ndarray] = {}

    for estimator, best_config in best_configs.items():
        if not best_config:
            continue
        params = _prepare_params(
            estimator,
            best_config,
            config["fit_kwargs_by_estimator"].get(estimator, {}),
            int(config["seed"]),
        )
        model, predictions = _train_estimator(
            estimator,
            params,
            monotone_vector,
            X_train,
            y_train,
            sample_weight_train,
            X_valid,
            y_valid,
            sample_weight_valid,
        )
        if predictions is None:
            continue
        predictions = np.clip(predictions, 0.0, None)
        alias = ESTIMATOR_ALIAS.get(estimator, estimator)
        metrics = _compute_metrics(y_valid, predictions)
        metrics_payload[alias] = metrics
        predictions_payload[alias] = predictions

        model_path = models_dir / f"best_model_holdout_{alias}.{ 'json' if estimator == 'xgboost' else 'txt' }"
        _save_model(estimator, model, model_path)
        importance_path = reports_dir / f"holdout_feature_importance_{alias}.csv"
        _save_feature_importance(estimator, model, feature_cols, importance_path)

        prediction_frame = _prepare_predictions_frame(
            valid_slice, predictions, sample_weight_valid, alias
        )
        prediction_frame.to_parquet(
            reports_dir / f"predictions_holdout_{alias}.parquet", index=False
        )

    if predictions_payload:
        ensemble_pred = np.mean(list(predictions_payload.values()), axis=0)
        metrics_payload["ensemble"] = _compute_metrics(y_valid, ensemble_pred)
        ensemble_frame = _prepare_predictions_frame(
            valid_slice, ensemble_pred, sample_weight_valid, "ensemble"
        )
        ensemble_frame.to_parquet(
            reports_dir / "predictions_holdout_ensemble.parquet", index=False
        )

    status = "ok"
    for entry in metrics_payload.values():
        if entry["competition_score"] < 0.70 or entry["MAPE"] > 0.45:
            status = "warning"
            break

    payload = {
        "status": status,
        "metrics": metrics_payload,
        "projection_drift_report": str(reports_dir / "projection_drift_202407.json"),
    }
    with (reports_dir / "holdout_metrics_v3.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    return payload


def _aggregate_top_features(
    importances: Iterable[pd.DataFrame], reports_dir: Path
) -> None:
    combined: Dict[str, float] = {}
    for df in importances:
        if df is None or df.empty:
            continue
        total = df["importance"].sum()
        if total <= 0:
            continue
        for _, row in df.iterrows():
            combined[row["feature"]] = combined.get(row["feature"], 0.0) + row["importance"] / total
    if not combined:
        return
    ranked = sorted(combined.items(), key=lambda item: item[1], reverse=True)
    top_features = [name for name, _ in ranked[:50]]
    reports_dir.mkdir(parents=True, exist_ok=True)
    with (reports_dir / "top_features.txt").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(top_features))


def train_full_model(
    features_df: pd.DataFrame,
    best_configs: Dict[str, Dict[str, object]],
    models_dir: Path | str,
    reports_dir: Path | str,
    config: Dict[str, object],
) -> None:
    models_dir = Path(models_dir)
    reports_dir = Path(reports_dir)
    _ensure_directories(models_dir, reports_dir)

    df = features_df.copy()
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df["month"] = df["month"].dt.to_period("M").dt.to_timestamp()
    cutoff = pd.Timestamp("2024-07-31")
    train_slice = df[(df["month"] <= cutoff) & df["target"].notna()].copy()
    if train_slice.empty:
        raise ValueError("No training data available for full model")

    feature_cols = _select_feature_columns(df)
    monotone_vector = _build_monotone_constraints(feature_cols)

    X_train = train_slice[feature_cols]
    y_train = train_slice["target"].astype(float).to_numpy()
    sample_weight_train = _compute_sample_weight(train_slice["month"])

    importances: List[pd.DataFrame] = []

    for estimator, best_config in best_configs.items():
        if not best_config:
            continue
        params = _prepare_params(
            estimator,
            best_config,
            config["fit_kwargs_by_estimator"].get(estimator, {}),
            int(config["seed"]),
        )
        model, _ = _train_estimator(
            estimator,
            params,
            monotone_vector,
            X_train,
            y_train,
            sample_weight_train,
        )
        alias = ESTIMATOR_ALIAS.get(estimator, estimator)
        model_path = models_dir / f"best_model_full_{alias}.{ 'json' if estimator == 'xgboost' else 'txt' }"
        _save_model(estimator, model, model_path)
        importance_path = reports_dir / f"feature_importance_full_{alias}.csv"
        importance_df = _save_feature_importance(estimator, model, feature_cols, importance_path)
        importances.append(importance_df)

    _aggregate_top_features(importances, reports_dir)


def run_training_pipeline(
    features_df: pd.DataFrame,
    config_path: Path | str,
    models_dir: Path | str = Path("models_v3"),
    reports_dir: Path | str = Path("reports_v3"),
    logs_dir: Path | str = Path("logs_v3"),
) -> Dict[str, object]:
    fold_results, summary = run_cross_validation(
        features_df,
        config_path=config_path,
        models_dir=models_dir,
        reports_dir=reports_dir,
        logs_dir=logs_dir,
    )
    if not fold_results:
        raise RuntimeError("Cross-validation did not produce any fold results")

    config = _load_config(Path(config_path))
    best_configs = {
        estimator: summary.get(estimator, {}).get("best_config") for estimator in config["estimator_list"]
    }

    holdout_metrics = train_holdout_model(
        features_df,
        best_configs,
        models_dir=models_dir,
        reports_dir=reports_dir,
        config=config,
    )
    train_full_model(
        features_df,
        best_configs,
        models_dir=models_dir,
        reports_dir=reports_dir,
        config=config,
    )

    return {
        "fold_results": [result.metrics for result in fold_results],
        "summary": summary,
        "holdout_metrics": holdout_metrics,
    }
