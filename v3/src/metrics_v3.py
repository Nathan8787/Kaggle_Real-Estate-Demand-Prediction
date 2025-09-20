from __future__ import annotations

import numpy as np

__all__ = ["competition_score", "competition_metric"]


def competition_score(y_true_raw, y_pred_raw) -> float:
    """Compute the v3 competition score capped between 0 and 1."""

    eps = 1e-9
    y_true = np.asarray(y_true_raw, dtype=float)
    y_pred = np.asarray(y_pred_raw, dtype=float)
    mask = ~np.isnan(y_true)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return 0.0

    ape = np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))
    if np.isnan(ape).all():
        return 0.0

    if (ape > 1.0).mean() > 0.30:
        return 0.0

    valid_mask = ape <= 1.0
    if valid_mask.sum() == 0:
        return 0.0

    scaled_mape = ape[valid_mask].mean() / valid_mask.mean()
    score = 1.0 - scaled_mape
    return float(np.clip(score, 0.0, 1.0))


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
    """FLAML-compatible wrapper returning the loss and diagnostic score."""

    y_pred_log = estimator.predict(X_val)
    y_true = np.clip(np.expm1(np.asarray(y_val, dtype=float)), 0.0, None)
    y_pred = np.clip(np.expm1(y_pred_log), 0.0, None)
    score = competition_score(y_true, y_pred)
    loss = 1.0 - score
    return loss, {"competition_score": score}
