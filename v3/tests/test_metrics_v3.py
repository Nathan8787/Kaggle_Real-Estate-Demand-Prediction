from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics_v3 import competition_metric, competition_score


class DummyEstimator:
    def __init__(self, predictions):
        self.predictions = np.asarray(predictions, dtype=float)

    def predict(self, X):
        return self.predictions


def _load_target_series(n_rows: int = 8) -> np.ndarray:
    data_path = ROOT.parent / "train" / "new_house_transactions.csv"
    df = pd.read_csv(data_path, encoding="utf-8-sig")
    subset = df.loc[df["sector"].eq("sector 1")].head(n_rows)
    return subset["amount_new_house_transactions"].astype(float).to_numpy()


def test_competition_metric_uses_raw_predictions():
    y_true = _load_target_series()
    y_pred = y_true * 1.02  # slightly optimistic yet realistic adjustment
    estimator = DummyEstimator(y_pred)
    loss, info = competition_metric(None, y_true, estimator, None, None, None)
    assert 0.0 <= info["competition_score"] <= 1.0
    assert np.isclose(loss, 1.0 - info["competition_score"], atol=1e-6)


def test_competition_score_penalizes_large_errors():
    y_true = _load_target_series()
    huge_pred = y_true * 25.0
    score = competition_score(y_true, huge_pred)
    assert score == 0.0
