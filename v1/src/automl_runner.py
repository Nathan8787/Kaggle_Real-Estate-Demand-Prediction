from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from flaml import AutoML
from sklearn.model_selection import BaseCrossValidator
from sklearn.preprocessing import StandardScaler

from .metrics import competition_metric, competition_score


@dataclass
class FoldResult:
    fold: int
    best_estimator: str
    best_iteration: int | None
    validation_score: float


class SingleFoldSplitter(BaseCrossValidator):
    def __init__(self, train_idx: np.ndarray, valid_idx: np.ndarray):
        self.train_idx = np.asarray(train_idx)
        self.valid_idx = np.asarray(valid_idx)

    def split(self, X=None, y=None, groups=None):
        yield self.train_idx, self.valid_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1


def _load_config(config_path: Path | str) -> Dict:
    import yaml

    config_path = Path(config_path)
    with config_path.open('r', encoding='utf-8') as fp:
        return yaml.safe_load(fp)


def _prepare_features(features_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    exclude_cols = {'id', 'sector', 'month', 'target', 'target_filled', 'is_future'}
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]
    X = features_df[feature_cols].astype(float)
    y = features_df['target'].astype(float)
    return X, y, feature_cols


def _write_scaler_report(output_dir: Path, fold_id: int, scaler: StandardScaler) -> None:
    payload = {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f'scaler_fold_{fold_id}.json').open('w', encoding='utf-8') as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def run(features_df: pd.DataFrame, folds: List[Tuple[np.ndarray, np.ndarray]], config_path: Path | str) -> Dict[str, List[str]]:
    config_path = Path(config_path)
    config = _load_config(config_path)
    if config.get('metric') != 'competition_score':
        raise ValueError('Config metric must be set to competition_score to align with specification.')

    config_base = config_path.parent
    logs_path = (config_base / config['log_file']).resolve()
    logs_path.parent.mkdir(parents=True, exist_ok=True)

    train_cutoff = pd.Timestamp('2024-07-31')
    train_df = features_df[features_df['month'] <= train_cutoff].copy()

    X_full, y_full, feature_cols = _prepare_features(train_df)
    y_full_array = y_full.to_numpy()
    y_full_log = np.log1p(y_full_array)

    fold_predictions: List[pd.DataFrame] = []
    fold_summaries: List[FoldResult] = []
    automl_objects: List[AutoML] = []

    reports_dir = Path('reports')
    reports_dir.mkdir(parents=True, exist_ok=True)

    estimator_list = config['estimator_list']
    removed_estimators = [est for est in estimator_list if est in {'lrl1', 'lrl2'}]
    estimator_list = [est for est in estimator_list if est not in {'lrl1', 'lrl2'}]

    for fold_id, (train_idx_global, valid_idx_global) in enumerate(folds, start=1):
        train_idx_global = np.asarray(train_idx_global)
        valid_idx_global = np.asarray(valid_idx_global)
        fold_indices = np.concatenate([train_idx_global, valid_idx_global])
        fold_subset = train_df.loc[fold_indices].sort_values(['month', 'sector_id']).reset_index()
        original_index = fold_subset['index'].to_numpy()

        fold_X = X_full.loc[original_index, :].reset_index(drop=True)
        fold_y = y_full.loc[original_index].reset_index(drop=True)
        fold_y_log = np.log1p(fold_y)

        train_mask = pd.Index(original_index).isin(train_idx_global)
        valid_mask = pd.Index(original_index).isin(valid_idx_global)
        train_positions = np.where(train_mask)[0]
        valid_positions = np.where(valid_mask)[0]

        scaler = StandardScaler()
        scaler.fit(fold_X.iloc[train_positions])
        scaled_values = scaler.transform(fold_X)
        fold_X_scaled = pd.DataFrame(scaled_values, columns=feature_cols)

        _write_scaler_report(reports_dir, fold_id, scaler)

        splitter = SingleFoldSplitter(train_positions, valid_positions)
        automl = AutoML()
        automl.fit(
            X_train=fold_X_scaled,
            y_train=fold_y_log,
            task='regression',
            metric=competition_metric,
            estimator_list=estimator_list,
            time_budget=config['time_budget'],
            log_file_name=str(logs_path),
            seed=config['seed'],
            eval_method='cv',
            split_type=splitter,
            n_jobs=config['n_jobs'],
        )

        y_valid_pred_log = automl.predict(fold_X_scaled.iloc[valid_positions])
        y_valid_pred = np.expm1(y_valid_pred_log)
        y_valid_true = fold_y.iloc[valid_positions].to_numpy()
        score = competition_score(y_valid_true, y_valid_pred)

        fold_predictions.append(
            pd.DataFrame(
                {
                    'fold': fold_id,
                    'id': train_df.loc[valid_idx_global, 'id'].values,
                    'month': train_df.loc[valid_idx_global, 'month'].values,
                    'sector_id': train_df.loc[valid_idx_global, 'sector_id'].values,
                    'target': y_valid_true,
                    'prediction': y_valid_pred,
                }
            )
        )

        fold_summaries.append(
            FoldResult(
                fold=fold_id,
                best_estimator=automl.best_estimator,
                best_iteration=automl.best_iteration,
                validation_score=float(score),
            )
        )
        automl_objects.append(automl)

    results_df = pd.concat(fold_predictions, axis=0, ignore_index=True)
    fold_score_map = {summary.fold: summary.validation_score for summary in fold_summaries}
    results_df['fold_score'] = results_df['fold'].map(fold_score_map)
    results_df.to_csv(reports_dir / 'fold_results.csv', index=False)

    scores_by_estimator: Dict[str, List[float]] = {}
    for summary in fold_summaries:
        scores_by_estimator.setdefault(summary.best_estimator, []).append(summary.validation_score)
    best_estimator = max(scores_by_estimator.items(), key=lambda item: np.mean(item[1]))[0]

    best_automl_idx = int(np.argmax([summary.validation_score for summary in fold_summaries]))
    best_automl = automl_objects[best_automl_idx]

    global_scaler = StandardScaler()
    global_scaler.fit(X_full)
    X_full_scaled = global_scaler.transform(X_full)

    final_model = best_automl.model
    final_model.fit(X_full_scaled, y_full_log)

    models_dir = Path('models')
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(global_scaler, models_dir / 'scaler.pkl')
    joblib.dump(final_model, models_dir / 'best_model.pkl')

    summary_payload = {
        'folds': [summary.__dict__ for summary in fold_summaries],
        'best_estimator': best_estimator,
        'best_fold': fold_summaries[best_automl_idx].fold,
        'feature_columns': feature_cols,
        'removed_estimators': removed_estimators,
        'target_transform': 'log1p_expm1',
    }
    with (reports_dir / 'automl_summary.json').open('w', encoding='utf-8') as fp:
        json.dump(summary_payload, fp, ensure_ascii=False, indent=2)

    log_entry = {
        'best_estimator': best_estimator,
        'fold_scores': {summary.fold: summary.validation_score for summary in fold_summaries},
    }
    with logs_path.open('a', encoding='utf-8') as log_fp:
        log_fp.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

    return summary_payload
