"""Unit tests for the competition scoring function."""

import numpy as np

from EDA.run_eda import competition_score


np.random.seed(42)


def test_perfect_score() -> None:
    y_true = np.array([100, 200, 300, 400], dtype=float)
    y_pred = np.array([100, 200, 300, 400], dtype=float)
    assert competition_score(y_true, y_pred) == 1.0


def test_single_ape_one() -> None:
    y_true = np.array([100, 200, 300, 400], dtype=float)
    y_pred = np.array([0, 200, 300, 400], dtype=float)
    score = competition_score(y_true, y_pred)
    assert np.isclose(score, 0.75)


def test_all_bad_predictions() -> None:
    y_true = np.array([100, 200, 300, 400], dtype=float)
    y_pred = np.zeros_like(y_true)
    assert competition_score(y_true, y_pred) == 0.0


def test_scaled_mape() -> None:
    y_true = np.array([100, 200, 300, 400], dtype=float)
    y_pred = np.array([150, 210, 330, 410], dtype=float)
    expected = 1 - np.mean([0.5, 0.05, 0.1, 0.025])
    score = competition_score(y_true, y_pred)
    assert np.isclose(score, expected)


def test_excessive_large_errors() -> None:
    """Ensure the penalty stage returns zero when large errors exceed 30%."""

    y_true = np.array([100, 200, 300, 400], dtype=float)
    # 註解：設計兩筆以上 APE>1 以觸發第一階段懲罰
    y_pred = np.array([50, 450, 700, 380], dtype=float)
    assert competition_score(y_true, y_pred) == 0.0
