from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.metrics import competition_score
from src.splits import generate_time_series_folds

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

CUTOFF_MONTH = pd.Timestamp('2024-07-31')
FEATURES_PATH = Path('data/processed/features.parquet')

def main() -> None:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f'Missing features file: {FEATURES_PATH}')

    features_df = pd.read_parquet(FEATURES_PATH)
    train_df = features_df[features_df['month'] <= CUTOFF_MONTH].copy()

    folds = generate_time_series_folds(train_df)
    drop_cols = {'id', 'sector', 'month', 'target', 'target_filled', 'is_future'}
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    scores = []
    for fold_id, (train_idx, valid_idx) in enumerate(folds, start=1):
        X_train = train_df.loc[train_idx, feature_cols].to_numpy(dtype=np.float64, copy=False)
        y_train = train_df.loc[train_idx, 'target'].to_numpy(dtype=np.float64, copy=False)
        y_train_log = np.log1p(y_train)
        X_valid = train_df.loc[valid_idx, feature_cols].to_numpy(dtype=np.float64, copy=False)
        y_valid = train_df.loc[valid_idx, 'target'].to_numpy(dtype=np.float64, copy=False)

        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0, ddof=0)
        std[std == 0.0] = 1.0
        X_train_scaled = (X_train - mean) / std
        X_valid_scaled = (X_valid - mean) / std

        model = XGBRegressor(**BEST_PARAMS)
        model.fit(X_train_scaled, y_train_log)
        y_pred_log = model.predict(X_valid_scaled)
        y_pred = np.expm1(y_pred_log)
        y_pred = np.clip(y_pred, 0.0, None)

        score = competition_score(y_valid, y_pred)
        scores.append(score)
        print(f'Fold {fold_id}: competition_score = {score:.6f}')

    print('\nMean competition_score:', np.mean(scores))

if __name__ == '__main__':
    main()
