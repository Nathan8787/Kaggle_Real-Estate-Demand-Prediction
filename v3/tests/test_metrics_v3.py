import numpy as np
from src.metrics_v2 import competition_metric, competition_score


class DummyEstimator:
    def __init__(self, preds_log):
        self.preds_log = np.array(preds_log)

    def predict(self, X):
        return self.preds_log


def test_competition_metric_log_restore():
    y_true = np.array([100.0, 120.0, 80.0])
    y_log = np.log1p(y_true)
    estimator = DummyEstimator(np.log1p([102.0, 118.0, 79.0]))
    loss, info = competition_metric(None, y_log, estimator, None, None, None)
    assert info["competition_score"] > 0.0
    assert 0.0 <= loss <= 1.0


def test_competition_score_penalizes_large_errors():
    score = competition_score([1.0, 1.0, 1.0, 1.0], [500.0, 500.0, 500.0, 0.01])
    assert score == 0.0
