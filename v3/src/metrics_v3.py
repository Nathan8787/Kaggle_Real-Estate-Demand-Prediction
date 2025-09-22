from __future__ import annotations

from typing import Callable

import numpy as np

__all__ = [
    "competition_score",
    "competition_metric",
    "make_flaml_metric",
    "competition_score_df",
]


def _prepare_arrays(y_true_raw, y_pred_raw) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true_raw, dtype=np.float64)
    y_pred = np.asarray(y_pred_raw, dtype=np.float64)
    mask = ~np.isnan(y_true)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    return y_true, np.clip(y_pred, 0.0, None)


def competition_score(y_true_raw, y_pred_raw) -> float:
    """Compute the official competition score following the v3.1 spec."""

    y_true, y_pred = _prepare_arrays(y_true_raw, y_pred_raw)
    if y_true.size == 0:
        return 0.0

    denominator = np.where(y_true > 0.0, y_true, 1.0)
    ape = np.abs(y_true - y_pred) / denominator
    contributions = np.where(ape <= 1.0, 1.0 - ape, 0.0)
    score = float(np.mean(contributions))
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

    predictions = estimator.predict(X_val)
    score = competition_score(y_val, predictions)
    loss = 1.0 - score
    return loss, {"competition_score": score}


def make_flaml_metric() -> Callable:
    """Return a FLAML metric callback minimising ``1 - competition_score``."""

    def flaml_metric(y_pred, dtrain):
        y_true = dtrain.get_label()
        score = competition_score(y_true, y_pred)
        return 1.0 - score

    return flaml_metric


def competition_score_df(df, pred_col: str, target_col: str) -> float:
    """Convenience helper for computing the score from a DataFrame."""

    if pred_col not in df.columns or target_col not in df.columns:
        raise KeyError("DataFrame must contain prediction and target columns")
    return competition_score(df[target_col].to_numpy(), df[pred_col].to_numpy())
