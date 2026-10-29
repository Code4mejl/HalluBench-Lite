"""Tests for metric computation."""

import numpy as np

from hallubench.metrics import compute_binary_metrics, explain_confusion, safe_metric
from sklearn.metrics import accuracy_score


def test_perfect_predictions():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    m = compute_binary_metrics(y_true, y_pred)
    assert m["accuracy"] == 1.0
    assert m["f1"] == 1.0
    assert m["n_evaluated"] == 4
    assert m["confusion_matrix"] == [[2, 0], [0, 2]]


def test_empty_predictions():
    m = compute_binary_metrics([], [])
    assert m["n_evaluated"] == 0
    assert m["accuracy"] == 0.0


def test_all_one_class():
    y_true = [0, 0, 0]
    y_pred = [0, 0, 1]
    m = compute_binary_metrics(y_true, y_pred)
    assert m["n_evaluated"] == 3
    assert 0.0 <= m["accuracy"] <= 1.0


def test_explain_confusion():
    cm = [[3, 1], [2, 4]]
    exp = explain_confusion(cm)
    assert "TN=3" in exp["true_negatives"]
    assert "FP=1" in exp["false_positives"]
    assert "FN=2" in exp["false_negatives"]
    assert "TP=4" in exp["true_positives"]


def test_safe_metric():
    assert safe_metric(accuracy_score, [], []) == 0.0
