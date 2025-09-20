from __future__ import annotations

import numpy as np


def competition_score(y_true, y_pred):
    eps = 1e-9
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ape = np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))
    if (ape > 1.0).mean() > 0.30:
        return 0.0
    mask = ape <= 1.0
    if mask.sum() == 0:
        return 0.0
    scaled_mape = ape[mask].mean() / mask.mean()
    return 1.0 - scaled_mape


def competition_metric(
    X_val,
    y_val,
    estimator,
    labels,
    X_train,
    y_train,
    weight_val=None,
    weight_train=None,
    *args,
    **kwargs,
):
    y_pred_log = estimator.predict(X_val)
    y_true_log = np.asarray(y_val, dtype=float)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_true_log)
    y_pred = np.clip(y_pred, 0.0, None)
    score = competition_score(y_true, y_pred)
    loss = 1.0 - score
    return loss, {'competition_score': score}
