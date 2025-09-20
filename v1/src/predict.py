from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd

from .features import build_feature_matrix


def _load_panel(panel_path: Path) -> pd.DataFrame:
    panel = pd.read_parquet(panel_path)
    panel['month'] = pd.to_datetime(panel['month'])
    return panel


def generate_submission(
    model_path: Path | str,
    scaler_path: Path | str,
    features_path: Path | str,
    sample_submission_path: Path | str,
    output_path: Path | str,
    summary_path: Path | str = Path('reports/automl_summary.json'),
    panel_path: Path | str = Path('data/processed/panel.parquet'),
) -> pd.DataFrame:
    model_path = Path(model_path)
    scaler_path = Path(scaler_path)
    sample_submission_path = Path(sample_submission_path)
    output_path = Path(output_path)
    summary_path = Path(summary_path)
    panel_path = Path(panel_path)

    if not summary_path.exists():
        raise FileNotFoundError('Missing automl summary file required for feature metadata.')
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    feature_columns: Sequence[str] = summary['feature_columns']

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    panel_df = _load_panel(panel_path)
    cutoff_month = panel_df.loc[panel_df['is_future'] == False, 'month'].max()
    forecast_months = sorted(panel_df.loc[panel_df['is_future'], 'month'].unique())

    predictions = []
    working_panel = panel_df.copy()

    for month in forecast_months:
        features_df, _ = build_feature_matrix(working_panel, features_path=None, reports_dir=None)
        current_features = features_df[features_df['month'] == month].copy()
        if current_features.empty:
            raise ValueError(f'No features generated for month {month}')
        X_iter = current_features.loc[:, feature_columns].astype(float)
        X_scaled = scaler.transform(X_iter)
        preds_log = model.predict(X_scaled)
        preds = np.expm1(preds_log)
        preds = np.clip(preds, 0.0, None)
        predictions.append(pd.DataFrame({
            'id': current_features['id'],
            'new_house_transaction_amount': preds,
        }))
        working_panel.loc[working_panel['month'] == month, 'target_filled'] = preds
        working_panel.loc[working_panel['month'] == month, 'target'] = preds

    predictions_df = pd.concat(predictions, axis=0, ignore_index=True)

    sample = pd.read_csv(sample_submission_path)
    submission = sample[['id']].merge(predictions_df, on='id', how='left')
    if submission['new_house_transaction_amount'].isna().any():
        missing_ids = submission[submission['new_house_transaction_amount'].isna()]['id'].tolist()
        raise ValueError(f'Missing predictions for ids: {missing_ids}')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    stats = {
        'prediction_count': int(len(submission)),
        'prediction_mean': float(np.mean(submission['new_house_transaction_amount'])),
        'prediction_std': float(np.std(submission['new_house_transaction_amount'])),
        'prediction_min': float(np.min(submission['new_house_transaction_amount'])),
        'prediction_max': float(np.max(submission['new_house_transaction_amount'])),
    }
    reports_dir = Path('reports')
    reports_dir.mkdir(parents=True, exist_ok=True)
    with (reports_dir / 'prediction_summary.json').open('w', encoding='utf-8') as fp:
        json.dump(stats, fp, ensure_ascii=False, indent=2)

    return submission
