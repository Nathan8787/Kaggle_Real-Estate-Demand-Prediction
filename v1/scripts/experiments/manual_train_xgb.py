from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# Paths
FEATURES_PATH = Path('data/processed/features.parquet')
MODEL_PATH = Path('models/best_model.pkl')
SCALER_PATH = Path('models/scaler.pkl')
CUTOFF_MONTH = pd.Timestamp('2024-07-31')

# Best parameters extracted from previous AutoML run (record_id 36)
BEST_PARAMS = {
    'n_estimators': 2898,
    'max_leaves': 4626,
    'min_child_weight': 0.035659797410131006,
    'learning_rate': 0.00664207402600315,
    'subsample': 0.3338656836252918,
    'colsample_bylevel': 1.0,
    'colsample_bytree': 0.9204947268747584,
    'reg_alpha': 17.269706552570955,
    'reg_lambda': 0.0009765625,
    'n_jobs': -1,
    'random_state': 42,
    'tree_method': 'hist',
    'grow_policy': 'lossguide',
}


def main() -> None:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f'Missing features file: {FEATURES_PATH}')

    features_df = pd.read_parquet(FEATURES_PATH)
    train_df = features_df[features_df['month'] <= CUTOFF_MONTH].copy()

    drop_cols = {'id', 'sector', 'month', 'target', 'target_filled', 'is_future'}
    feature_cols = [col for col in train_df.columns if col not in drop_cols]

    X = train_df[feature_cols].to_numpy(dtype=np.float64, copy=False)
    y_raw = train_df['target'].to_numpy(dtype=np.float64, copy=False)
    y_log = np.log1p(y_raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = XGBRegressor(**BEST_PARAMS)
    model.fit(X_scaled, y_log)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f'Model saved to {MODEL_PATH}')
    print(f'Scaler saved to {SCALER_PATH}')
    summary_payload = {
        'folds': [],
        'best_estimator': 'xgboost_manual',
        'best_fold': None,
        'feature_columns': feature_cols,
        'removed_estimators': ['lrl1', 'lrl2'],
        'target_transform': 'log1p_expm1',
    }
    reports_dir = Path('reports')
    reports_dir.mkdir(parents=True, exist_ok=True)
    with (reports_dir / 'automl_summary.json').open('w', encoding='utf-8') as fp:
        import json
        json.dump(summary_payload, fp, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
