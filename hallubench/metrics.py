"""Evaluation metrics, confusion matrices, and latency helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from hallubench import LABEL_HALLUCINATED, LABEL_SUPPORTED, POSITIVE_CLASS


def safe_metric(func, y_true, y_pred, **kwargs) -> float:
    """Call a sklearn metric function; return 0.0 on ValueError (e.g. empty)."""
    try:
        if len(y_true) == 0:
            return 0.0
        val = func(y_true, y_pred, **kwargs)
        if isinstance(val, (np.floating, float)):
            return float(val)
        return val
    except (ValueError, ZeroDivisionError):
        return 0.0


def compute_binary_metrics(
    y_true: List[int] | np.ndarray,
    y_pred: List[int] | np.ndarray,
    labels: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Compute accuracy, precision, recall, F1, macro-F1, and confusion matrix.

    Positive class is hallucinated (1). Abstentions should already be filtered out.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if labels is None:
        labels = [LABEL_SUPPORTED, LABEL_HALLUCINATED]

    n = len(y_true)
    if n == 0:
        return {
            "n_evaluated": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "macro_f1": 0.0,
            "confusion_matrix": [[0, 0], [0, 0]],
            "support": {0: 0, 1: 0},
            "classification_report": {},
        }

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # Ensure 2x2
    if cm.shape != (2, 2):
        full = np.zeros((2, 2), dtype=int)
        for i, li in enumerate(labels):
            for j, lj in enumerate(labels):
                if i < cm.shape[0] and j < cm.shape[1]:
                    full[i, j] = cm[i, j]
        cm = full

    support = {
        LABEL_SUPPORTED: int((y_true == LABEL_SUPPORTED).sum()),
        LABEL_HALLUCINATED: int((y_true == LABEL_HALLUCINATED).sum()),
    }

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=["supported", "hallucinated"],
        output_dict=True,
        zero_division=0,
    )

    return {
        "n_evaluated": int(n),
        "accuracy": safe_metric(accuracy_score, y_true, y_pred),
        "precision": safe_metric(
            precision_score, y_true, y_pred, pos_label=POSITIVE_CLASS, zero_division=0
        ),
        "recall": safe_metric(
            recall_score, y_true, y_pred, pos_label=POSITIVE_CLASS, zero_division=0
        ),
        "f1": safe_metric(f1_score, y_true, y_pred, pos_label=POSITIVE_CLASS, zero_division=0),
        "macro_f1": safe_metric(f1_score, y_true, y_pred, average="macro", zero_division=0),
        "confusion_matrix": cm.tolist(),
        "support": support,
        "classification_report": report,
    }


def explain_confusion(cm: List[List[int]]) -> Dict[str, str]:
    """Plain-English explanation of TP/TN/FP/FN for hallucinated-positive framing."""
    # cm rows = true, cols = predicted; index 0 = supported, 1 = hallucinated
    tn = cm[0][0] if len(cm) > 0 and len(cm[0]) > 0 else 0
    fp = cm[0][1] if len(cm) > 0 and len(cm[0]) > 1 else 0
    fn = cm[1][0] if len(cm) > 1 and len(cm[1]) > 0 else 0
    tp = cm[1][1] if len(cm) > 1 and len(cm[1]) > 1 else 0
    return {
        "true_negatives": (
            f"TN={tn}: correctly identified as supported (non-hallucinated)."
        ),
        "false_positives": (
            f"FP={fp}: predicted hallucinated but actually supported — false alarm."
        ),
        "false_negatives": (
            f"FN={fn}: predicted supported but actually hallucinated — missed hallucination."
        ),
        "true_positives": (
            f"TP={tp}: correctly identified as hallucinated/unsupported."
        ),
    }


def aggregate_latency(latencies_ms: List[float]) -> Dict[str, float]:
    """Summary statistics for per-example latencies in milliseconds."""
    if not latencies_ms:
        return {"mean_ms": 0.0, "median_ms": 0.0, "total_ms": 0.0, "n": 0}
    arr = np.asarray(latencies_ms, dtype=float)
    return {
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "total_ms": float(np.sum(arr)),
        "n": int(len(arr)),
    }


def error_type_breakdown(
    df: pd.DataFrame,
    pred_col: str,
    label_col: str = "label",
    error_col: str = "error_type",
) -> pd.DataFrame:
    """Count correct/incorrect predictions per error_type annotation."""
    if error_col not in df.columns or pred_col not in df.columns:
        return pd.DataFrame()
    work = df[[label_col, pred_col, error_col]].copy()
    work["correct"] = work[label_col] == work[pred_col]
    grouped = (
        work.groupby(error_col)
        .agg(total=("correct", "size"), correct=("correct", "sum"))
        .reset_index()
    )
    grouped["incorrect"] = grouped["total"] - grouped["correct"]
    grouped["accuracy"] = grouped.apply(
        lambda r: r["correct"] / r["total"] if r["total"] else 0.0, axis=1
    )
    return grouped
