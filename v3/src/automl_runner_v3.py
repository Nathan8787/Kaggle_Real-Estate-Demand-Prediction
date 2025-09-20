from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from flaml import AutoML
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .metrics_v3 import competition_metric, competition_score
from .splits_v3 import FOLD_DEFINITIONS_V3, HOLDOUT_WINDOW, generate_time_series_folds_v3

logger = logging.getLogger(__name__)

__all__ = ["run_cross_validation", "train_holdout_model", "train_full_model", "run_training_pipeline"]


@dataclass
class FoldResult:
    fold_id: int
    config: Dict[str, object]
    best_iteration: Optional[int]
    metrics: Dict[str, float]
    scaler_path: Path
    model_path: Path


class _ProgressPrinter:
    """Render a lightweight console progress bar for long-running stages."""

    def __init__(self, total: int, label: str = "Progress") -> None:
        self.total = max(int(total), 1)
        self.label = label
        self.current = 0
        self._last_length = 0

    def _render(self, message: str) -> None:
        ratio = self.current / self.total
        bar_size = 24
        filled = int(round(bar_size * ratio))
        bar = "█" * filled + "░" * (bar_size - filled)
        display = f"{self.label}: [{bar}] {self.current}/{self.total} {message}"
        padding = max(0, self._last_length - len(display))
        sys.stdout.write("\r" + display + " " * padding)
        sys.stdout.flush()
        self._last_length = len(display)

    def announce(self, message: str) -> None:
        """Update the status message without advancing progress."""

        self._render(message)

    def step(self, message: str) -> None:
        """Advance the progress bar and update the message."""

        self.current = min(self.current + 1, self.total)
        self._render(message)
        if self.current == self.total:
            sys.stdout.write("\n")
            sys.stdout.flush()

    def close(self) -> None:
        if self.current < self.total:
            self._render("stopped")
            sys.stdout.write("\n")
            sys.stdout.flush()


def _load_config(config_path: Path) -> Dict[str, object]:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
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
        "fit_kwargs",
        "search_space",
    }
    missing = required_keys - set(config)
    if missing:
        raise KeyError(f"Missing required keys in automl config: {sorted(missing)}")
    return config


def _ensure_directories(models_dir: Path, reports_dir: Path, logs_dir: Path) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)


def _select_feature_columns(features_df: pd.DataFrame) -> List[str]:
    exclude = {"month", "id", "target"}
    exclude = {"month", "id"}
    if "target" not in features_df.columns:
        raise KeyError("features_df must contain a 'target' column")
    feature_cols = [col for col in features_df.columns if col not in exclude]
    return feature_cols


def _split_float_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=["float32", "float64"]).columns.tolist()


def _resolve_metric(metric_setting) -> Tuple[object, str]:
    """Map configuration metric names to FLAML-compatible callables."""

    if isinstance(metric_setting, str):
        if metric_setting.lower() == "competition_score":
            return flaml_competition_metric, "competition_score"
        return metric_setting, metric_setting
    if callable(metric_setting):
        return metric_setting, getattr(metric_setting, "__name__", "custom_metric")
    return metric_setting, str(metric_setting)


def _register_metric(automl: AutoML, metric_param, metric_label: str):
    """Register custom metrics with FLAML when supported."""

    metric_for_fit = metric_param
    if callable(metric_param):
        if hasattr(automl, "add_metric"):
            kwargs = {}
            greater_flag = getattr(metric_param, "greater_is_better", None)
            if greater_flag is None and metric_label == "competition_score":
                greater_flag = True
            if greater_flag is not None:
                kwargs["greater_is_better"] = bool(greater_flag)
            try:
                automl.add_metric(metric_label, metric_param, **kwargs)
            except TypeError:
                automl.add_metric(metric_label, metric_param)
            metric_for_fit = metric_label
        else:
            logger.warning(
                "AutoML.add_metric not available; passing metric callable directly"
            )
    return metric_for_fit


def flaml_competition_metric(
    X_train,
    y_train,
    estimator,
    labels,
    X_val,
    y_val,
    weight_val=None,
    weight_train=None,
    *args,
    **kwargs,
):
    """Adapter registering the competition metric with FLAML."""

    loss, info = competition_metric(
        X_val=X_val,
        y_val=y_val,
        estimator=estimator,
        labels=labels,
        X_train=X_train,
        y_train=y_train,
        weight_val=weight_val,
        weight_train=weight_train,
    )
    return loss, info


flaml_competition_metric.greater_is_better = True


def _apply_scaler(
    scaler: StandardScaler,
    df: pd.DataFrame,
    float_cols: Sequence[str],
) -> pd.DataFrame:
    if not float_cols:
        return df
    transformed = df.copy()
    transformed.loc[:, float_cols] = scaler.transform(df[float_cols])
    return transformed


def _compute_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    numerator = np.abs(y_pred - y_true)
    denominator = np.abs(y_true) + np.abs(y_pred) + 1e-6
    return float(np.mean(2.0 * numerator / denominator))


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    error = y_pred - y_true
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(np.mean(error**2)))
    mape = float(np.mean(np.abs(error) / (np.abs(y_true) + 1e-6)))
    smape = _compute_smape(y_true, y_pred)
    score = competition_score(y_true, y_pred)
    r2 = r2_score(y_true, y_pred) if y_true.size > 1 else 0.0
    return {
        "competition_score": float(score),
        "SMAPE": float(smape),
        "MAPE": float(mape),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
    }


def _save_scaler_artifacts(
    scaler: StandardScaler,
    float_cols: Sequence[str],
    scaler_path: Path,
    report_path: Path,
) -> None:
    joblib.dump({"scaler": scaler, "float_cols": list(float_cols)}, scaler_path)
    stats = {
        "float_columns": list(float_cols),
        "mean": scaler.mean_.tolist() if float_cols else [],
        "scale": scaler.scale_.tolist() if float_cols else [],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)


def _log_trials(
    automl: AutoML,
    fold_id: int,
    logs_path: Path,
) -> None:
    trial_rows: List[Dict[str, object]] = []
    for iteration, payload in automl.config_history.items():
        estimator, config, wall_time = payload
        trial_rows.append(
            {
                "fold": fold_id,
                "iteration": iteration,
                "estimator": estimator,
                "wall_time": float(wall_time),
                "config": json.dumps(config),
            }
        )
    if not trial_rows:
        return
    logs_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(trial_rows)
    if logs_path.exists():
        df_existing = pd.read_csv(logs_path)
        df = pd.concat([df_existing, df], ignore_index=True)
    df.to_csv(logs_path, index=False)


def _merge_config_with_fit_kwargs(
    estimator_config: Dict[str, object],
    fit_kwargs: Dict[str, object],
    seed: int,
) -> Dict[str, object]:
    merged = {**estimator_config}
    merged.update(fit_kwargs)
    merged.setdefault("random_state", seed)
    merged.setdefault("objective", "reg:squarederror")
    return merged


def run_cross_validation(
    features_df: pd.DataFrame,
    config_path: Path | str,
    models_dir: Path | str,
    reports_dir: Path | str,
    logs_dir: Path | str,
) -> Tuple[List[FoldResult], Dict[str, object]]:
    config = _load_config(Path(config_path))
    models_dir = Path(models_dir)
    reports_dir = Path(reports_dir)
    logs_dir = Path(logs_dir)
    _ensure_directories(models_dir, reports_dir, logs_dir)

    ordered_df = features_df.copy()
    ordered_df["month"] = pd.to_datetime(ordered_df["month"], errors="coerce")

    folds = generate_time_series_folds_v3(ordered_df, reports_dir=reports_dir)
    feature_cols = _select_feature_columns(ordered_df)

    metric_param, metric_label = _resolve_metric(config["metric"])
    total_folds = len(folds)
    progress = _ProgressPrinter(total_folds, label="Cross-validation") if total_folds else None

    fold_metrics: List[Dict[str, object]] = []
    fold_results: List[FoldResult] = []
    fold_scores: List[float] = []
    best_overall_score = -np.inf
    best_overall_config: Optional[Dict[str, object]] = None

    for fold_id, (train_idx, valid_idx) in enumerate(folds, start=1):
        if progress:
            progress.announce(
                f"Fold {fold_id}/{total_folds}: preparing datasets"
            )
        train_slice = ordered_df.loc[train_idx].copy()
        valid_slice = ordered_df.loc[valid_idx].copy()

        train_slice = train_slice[train_slice["target"].notna()]
        valid_slice = valid_slice[valid_slice["target"].notna()]
        if train_slice.empty or valid_slice.empty:
            logger.warning("Fold %s skipped due to insufficient labeled data", fold_id)
            if progress:
                progress.step(f"Fold {fold_id} skipped")
            continue

        X_train = train_slice[feature_cols].copy()
        X_valid = valid_slice[feature_cols].copy()
        y_train_raw = train_slice["target"].astype(float).to_numpy()
        y_valid_raw = valid_slice["target"].astype(float).to_numpy()
        y_train_log = np.log1p(y_train_raw)
        y_valid_log = np.log1p(y_valid_raw)

        float_cols = _split_float_columns(X_train)
        scaler = StandardScaler(with_mean=True, with_std=True)
        if float_cols:
            scaler.fit(X_train[float_cols])
            X_train.loc[:, float_cols] = scaler.transform(X_train[float_cols])
            X_valid.loc[:, float_cols] = scaler.transform(X_valid[float_cols])
        scaler_path = models_dir / f"fold_{fold_id}_scaler.pkl"
        scaler_report_path = reports_dir / f"fold_{fold_id}_scaler_stats.json"
        _save_scaler_artifacts(scaler, float_cols, scaler_path, scaler_report_path)

        automl = AutoML()
        metric_for_fit = _register_metric(automl, metric_param, metric_label)
        if progress:
            progress.announce(
                f"Fold {fold_id}/{total_folds}: AutoML search"
            )
        logger.info(
            "Fold %s/%s: running AutoML search for %.0f seconds",
            fold_id,
            total_folds,
            float(config["time_budget_per_fold"]),
        )
        automl.fit(
            X_train=X_train.values,
            y_train=y_train_log,
            task=config["task"],
            metric=metric_for_fit,
            estimator_list=config["estimator_list"],
            time_budget=float(config["time_budget_per_fold"]),
            n_jobs=int(config["n_jobs"]),
            eval_method="holdout",
            X_val=X_valid.values,
            y_val=y_valid_log,
            verbose=0,
            fit_kwargs_by_estimator={"xgboost": config.get("fit_kwargs", {})},
            seed=int(config["seed"]),
        )
        best_iteration = getattr(automl, "best_iteration", None)
        min_trials = int(config["min_trials_per_fold"])
        if best_iteration is not None and best_iteration < min_trials:
            extra_budget = max(0.0, 1200.0 - float(config["time_budget_per_fold"]))
            if extra_budget > 0:
                logger.info(
                    "Fold %s completed %s trials; extending search with %.0f additional seconds",
                    fold_id,
                    best_iteration,
                    extra_budget,
                )
                if progress:
                    progress.announce(
                        f"Fold {fold_id}/{total_folds}: extending search"
                    )
                automl.fit(
                    X_train=X_train.values,
                    y_train=y_train_log,
                    task=config["task"],
                    metric=metric_for_fit,
                    estimator_list=config["estimator_list"],
                    time_budget=extra_budget,
                    n_jobs=int(config["n_jobs"]),
                    eval_method="holdout",
                    X_val=X_valid.values,
                    y_val=y_valid_log,
                    verbose=0,
                    fit_kwargs_by_estimator={"xgboost": config.get("fit_kwargs", {})},
                    seed=int(config["seed"]),
                )
                best_iteration = getattr(automl, "best_iteration", None)

        best_config = automl.best_config.copy()
        params = _merge_config_with_fit_kwargs(best_config, config.get("fit_kwargs", {}), int(config["seed"]))
        params.setdefault("n_estimators", best_config.get("n_estimators", 500))
        logger.info("Fold %s/%s: retraining best configuration", fold_id, total_folds)
        model = XGBRegressor(**params)
        model.fit(
            X_train.values,
            y_train_log,
            eval_set=[(X_valid.values, y_valid_log)],
            early_stopping_rounds=100,
            verbose=False,
        )

        model_path = models_dir / f"fold_{fold_id}_model.json"
        model.get_booster().save_model(model_path)

        y_pred_log = model.predict(X_valid.values)
        y_pred = np.clip(np.expm1(y_pred_log), 0.0, None)
        metrics = _compute_metrics(y_valid_raw, y_pred)

        train_slice = ordered_df.loc[train_idx].copy()
        valid_slice = ordered_df.loc[valid_idx].copy()

        train_slice = train_slice[train_slice["target"].notna()]
        valid_slice = valid_slice[valid_slice["target"].notna()]
        if train_slice.empty or valid_slice.empty:
            logger.warning("Fold %s skipped due to insufficient labeled data", fold_id)
            continue

        X_train = train_slice[feature_cols].copy()
        X_valid = valid_slice[feature_cols].copy()
        y_train_raw = train_slice["target"].astype(float).to_numpy()
        y_valid_raw = valid_slice["target"].astype(float).to_numpy()
        y_train_log = np.log1p(y_train_raw)
        y_valid_log = np.log1p(y_valid_raw)

        float_cols = _split_float_columns(X_train)
        scaler = StandardScaler(with_mean=True, with_std=True)
        if float_cols:
            scaler.fit(X_train[float_cols])
            X_train.loc[:, float_cols] = scaler.transform(X_train[float_cols])
            X_valid.loc[:, float_cols] = scaler.transform(X_valid[float_cols])
        scaler_path = models_dir / f"fold_{fold_id}_scaler.pkl"
        scaler_report_path = reports_dir / f"fold_{fold_id}_scaler_stats.json"
        _save_scaler_artifacts(scaler, float_cols, scaler_path, scaler_report_path)

        automl = AutoML()
        automl.fit(
            X_train=X_train.values,
            y_train=y_train_log,
            task=config["task"],
            metric=competition_metric,
            estimator_list=config["estimator_list"],
            time_budget=float(config["time_budget_per_fold"]),
            n_jobs=int(config["n_jobs"]),
            eval_method="holdout",
            X_val=X_valid.values,
            y_val=y_valid_log,
            verbose=0,
            fit_kwargs_by_estimator={"xgboost": config.get("fit_kwargs", {})},
            seed=int(config["seed"]),
        )
        best_iteration = getattr(automl, "best_iteration", None)
        if best_iteration is not None and best_iteration < int(config["min_trials_per_fold"]):
            logger.info(
                "Fold %s completed %s trials; extending search to meet minimum %s",
                fold_id,
                best_iteration,
                config["min_trials_per_fold"],
            )
            automl.fit(
                X_train=X_train.values,
                y_train=y_train_log,
                task=config["task"],
                metric=competition_metric,
                estimator_list=config["estimator_list"],
                time_budget=float(config["time_budget_per_fold"]),
                n_jobs=int(config["n_jobs"]),
                eval_method="holdout",
                X_val=X_valid.values,
                y_val=y_valid_log,
                verbose=0,
                fit_kwargs_by_estimator={"xgboost": config.get("fit_kwargs", {})},
                seed=int(config["seed"]),
            )
            best_iteration = getattr(automl, "best_iteration", None)

        best_config = automl.best_config.copy()
        params = _merge_config_with_fit_kwargs(best_config, config.get("fit_kwargs", {}), int(config["seed"]))
        params.setdefault("n_estimators", best_config.get("n_estimators", 500))
        model = XGBRegressor(**params)
        model.fit(
            X_train.values,
            y_train_log,
            eval_set=[(X_valid.values, y_valid_log)],
            early_stopping_rounds=100,
            verbose=False,
        )

        model_path = models_dir / f"fold_{fold_id}_model.json"
        model.get_booster().save_model(model_path)

        y_pred_log = model.predict(X_valid.values)
        y_pred = np.clip(np.expm1(y_pred_log), 0.0, None)
        metrics = _compute_metrics(y_valid_raw, y_pred)

        prediction_frame = pd.DataFrame(
            {
                "id": valid_slice.get("id"),
                "month": valid_slice["month"],
                "sector_id": valid_slice["sector_id"],
                "target": y_valid_raw,
                "prediction": y_pred,
                "target_available_flag": valid_slice.get("target_available_flag"),
                "target": y_valid_raw,
                "prediction": y_pred,
            }
        )
        prediction_path = reports_dir / f"predictions_fold_{fold_id}.parquet"
        prediction_frame.to_parquet(prediction_path, index=False)

        fold_metrics.append({"fold": fold_id, **metrics})
        fold_scores.append(metrics["competition_score"])
        fold_results.append(
            FoldResult(
                fold_id=fold_id,
                config=best_config,
                best_iteration=best_iteration,
                metrics=metrics,
                scaler_path=scaler_path,
                model_path=model_path,
            )
        )

        if metrics["competition_score"] > best_overall_score:
            best_overall_score = metrics["competition_score"]
            best_overall_config = best_config.copy()

        _log_trials(automl, fold_id, logs_dir / "automl_trials.csv")

        if progress:
            progress.step(f"Fold {fold_id} score={metrics['competition_score']:.3f}")

    if fold_metrics:
        metrics_df = pd.DataFrame(fold_metrics)
        metrics_df.to_csv(reports_dir / "fold_metrics_v3.csv", index=False)

    if fold_metrics:
        metrics_df = pd.DataFrame(fold_metrics)
        metrics_df.to_csv(reports_dir / "fold_metrics_v3.csv", index=False)

    summary = {
        "best_config": best_overall_config or {},
        "fold_scores": fold_scores,
        "avg_score": float(np.mean(fold_scores)) if fold_scores else None,
        "feature_count": len(feature_cols),
        "target_transform": "log1p_expm1",
        "metric": metric_label,
    }
    with (reports_dir / "automl_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    return fold_results, summary


def train_holdout_model(
    features_df: pd.DataFrame,
    best_config: Dict[str, object],
    models_dir: Path | str,
    reports_dir: Path | str,
    config: Dict[str, object],
) -> Dict[str, float]:
    models_dir = Path(models_dir)
    reports_dir = Path(reports_dir)
    _ensure_directories(models_dir, reports_dir, reports_dir)

    features_df = features_df.copy()
    features_df["month"] = pd.to_datetime(features_df["month"], errors="coerce")

    train_start, train_end, holdout_start, holdout_end = [
        pd.Timestamp(value) for value in HOLDOUT_WINDOW
    ]
    train_mask = (
        (features_df["month"] >= train_start)
        & (features_df["month"] <= train_end)
        & (features_df["target"].notna())
    )
    valid_mask = (
        (features_df["month"] >= holdout_start)
        & (features_df["month"] <= holdout_end)
        & (features_df["target"].notna())
    )

    train_slice = features_df.loc[train_mask].copy()
    valid_slice = features_df.loc[valid_mask].copy()
    if train_slice.empty or valid_slice.empty:
        raise ValueError("Holdout training requires labeled data up to 2024-07")

    feature_cols = _select_feature_columns(features_df)
    X_train = train_slice[feature_cols].copy()
    X_valid = valid_slice[feature_cols].copy()
    y_train_raw = train_slice["target"].astype(float).to_numpy()
    y_valid_raw = valid_slice["target"].astype(float).to_numpy()
    y_train_log = np.log1p(y_train_raw)
    y_valid_log = np.log1p(y_valid_raw)

    float_cols = _split_float_columns(X_train)
    scaler = StandardScaler(with_mean=True, with_std=True)
    if float_cols:
        scaler.fit(X_train[float_cols])
        X_train.loc[:, float_cols] = scaler.transform(X_train[float_cols])
        X_valid.loc[:, float_cols] = scaler.transform(X_valid[float_cols])
    _save_scaler_artifacts(
        scaler,
        float_cols,
        models_dir / "scaler_holdout.pkl",
        reports_dir / "fold_holdout_scaler_stats.json",
    )

    params = _merge_config_with_fit_kwargs(best_config, config.get("fit_kwargs", {}), int(config["seed"]))
    model = XGBRegressor(**params)
    model.fit(
        X_train.values,
        y_train_log,
        eval_set=[(X_valid.values, y_valid_log)],
        early_stopping_rounds=100,
        verbose=False,
    )
    model.get_booster().save_model(models_dir / "best_model_holdout.json")

    y_pred_log = model.predict(X_valid.values)
    y_pred = np.clip(np.expm1(y_pred_log), 0.0, None)
    metrics = _compute_metrics(y_valid_raw, y_pred)

    prediction_frame = valid_slice[["id", "month", "sector_id", "target", "target_available_flag"]].copy()
    prediction_frame["prediction"] = y_pred
    prediction_frame.to_parquet(reports_dir / "predictions_holdout.parquet", index=False)

    metrics_path = reports_dir / "holdout_metrics_v3.json"
    payload = {
        "metrics": metrics,
        "projection_drift_report": str(reports_dir / "projection_drift_202407.json"),
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    if metrics["competition_score"] < 0.70 or metrics["MAPE"] > 0.45:
        raise RuntimeError(
            "Holdout performance below threshold: "
            f"score={metrics['competition_score']:.3f}, MAPE={metrics['MAPE']:.3f}"
        )
    return metrics


def train_full_model(
    features_df: pd.DataFrame,
    best_config: Dict[str, object],
    models_dir: Path | str,
    reports_dir: Path | str,
    config: Dict[str, object],
) -> None:
    models_dir = Path(models_dir)
    reports_dir = Path(reports_dir)
    _ensure_directories(models_dir, reports_dir, reports_dir)

    features_df = features_df.copy()
    features_df["month"] = pd.to_datetime(features_df["month"], errors="coerce")

    cutoff = pd.Timestamp("2024-07-31")
    train_slice = features_df[
        (features_df["month"] <= cutoff) & (features_df["target"].notna())
    ].copy()
    if train_slice.empty:
        raise ValueError("Full training set is empty")

    feature_cols = _select_feature_columns(features_df)
    X_train = train_slice[feature_cols].copy()
    y_train_log = np.log1p(train_slice["target"].astype(float).to_numpy())

    float_cols = _split_float_columns(X_train)
    scaler = StandardScaler(with_mean=True, with_std=True)
    if float_cols:
        scaler.fit(X_train[float_cols])
        X_train.loc[:, float_cols] = scaler.transform(X_train[float_cols])
    _save_scaler_artifacts(
        scaler,
        float_cols,
        models_dir / "scaler_full.pkl",
        reports_dir / "fold_full_scaler_stats.json",
    )

    params = _merge_config_with_fit_kwargs(best_config, config.get("fit_kwargs", {}), int(config["seed"]))
    model = XGBRegressor(**params)
    model.fit(X_train.values, y_train_log, verbose=False)
    booster = model.get_booster()
    booster.save_model(models_dir / "best_model_full.json")

    importance = booster.get_score(importance_type="gain")
    if importance:
        importance_items = sorted(importance.items(), key=lambda item: item[1], reverse=True)
        df = pd.DataFrame(importance_items, columns=["feature", "gain"])
        df.to_csv(reports_dir / "feature_importance_gain.csv", index=False)
        top_features = [name for name, _ in importance_items[:50]]
        with (reports_dir / "top_features.txt").open("w", encoding="utf-8") as handle:
            handle.write("\n".join(top_features))


def run_training_pipeline(
    features_df: pd.DataFrame,
    config_path: Path | str,
    models_dir: Path | str = Path("models_v3"),
    reports_dir: Path | str = Path("reports_v3"),
    logs_dir: Path | str = Path("logs_v3"),
) -> Dict[str, object]:
    logger.info("Starting cross-validation pipeline")
    fold_results, summary = run_cross_validation(
        features_df,
        config_path=config_path,
        models_dir=models_dir,
        reports_dir=reports_dir,
        logs_dir=logs_dir,
    )
    if not fold_results:
        raise RuntimeError("No cross-validation folds were processed")

    best_config = summary.get("best_config") or fold_results[0].config
    config = _load_config(Path(config_path))

    logger.info("Training holdout model with best configuration")
    holdout_metrics = train_holdout_model(
        features_df,
        best_config,
        models_dir=models_dir,
        reports_dir=reports_dir,
        config=config,
    )
    logger.info(
        "Holdout performance: score=%.3f, MAPE=%.3f",
        holdout_metrics.get("competition_score", float("nan")),
        holdout_metrics.get("MAPE", float("nan")),
    )
    logger.info("Training full model on all available history")
    train_full_model(
        features_df,
        best_config,
        models_dir=models_dir,
        reports_dir=reports_dir,
        config=config,
    )

    return {
        "fold_results": [result.metrics for result in fold_results],
        "summary": summary,
        "holdout_metrics": holdout_metrics,
    }
