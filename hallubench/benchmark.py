"""Benchmark runner: evaluate selected detectors on a dataset."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from hallubench import LABEL_HALLUCINATED, LABEL_SUPPORTED
from hallubench.metrics import aggregate_latency, compute_binary_metrics, error_type_breakdown
from detectors import AVAILABLE_METHODS


def run_benchmark(
    df: pd.DataFrame,
    method_names: List[str],
    sample_limit: Optional[int] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    detector_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run selected detectors on the dataset and return results + metrics.

    Parameters
    ----------
    df : validated DataFrame with evidence, claim, label columns
    method_names : keys from AVAILABLE_METHODS
    sample_limit : optional max rows (head)
    progress_callback : optional fn(fraction_0_to_1, message)
    detector_kwargs : optional per-method init kwargs
    """
    detector_kwargs = detector_kwargs or {}
    work = df.copy()
    if sample_limit is not None and sample_limit > 0:
        work = work.head(int(sample_limit)).reset_index(drop=True)

    n = len(work)
    if n == 0:
        return {"error": "No examples to evaluate.", "results_df": work, "metrics": {}}

    evidences = work["evidence"].astype(str).tolist()
    claims = work["claim"].astype(str).tolist()
    y_true = work["label"].astype(int).tolist()

    results_df = work.copy()
    all_metrics: Dict[str, Any] = {}
    method_configs: Dict[str, Any] = {}
    total_start = time.perf_counter()

    valid_methods = [m for m in method_names if m in AVAILABLE_METHODS]
    if not valid_methods:
        return {"error": "No valid methods selected.", "results_df": results_df, "metrics": {}}

    n_methods = len(valid_methods)
    for mi, method_name in enumerate(valid_methods):
        if progress_callback:
            progress_callback(mi / n_methods, f"Initializing {method_name}...")

        cls = AVAILABLE_METHODS[method_name]
        kwargs = detector_kwargs.get(method_name, {})
        detector = cls(**kwargs)
        method_configs[method_name] = detector.get_config()

        preds: List[int] = []
        scores: List[float] = []
        abstained: List[bool] = []
        latencies: List[float] = []
        backends: List[str] = []
        details_list: List[Any] = []

        for i, (ev, cl) in enumerate(zip(evidences, claims)):
            if progress_callback:
                frac = (mi + (i + 1) / n) / n_methods
                progress_callback(frac, f"{method_name}: example {i + 1}/{n}")
            try:
                out = detector.predict_one(ev, cl)
            except RuntimeError as exc:
                # NLI unavailable — mark as failed for this method
                out = {
                    "prediction": -2,  # error sentinel
                    "score": 0.0,
                    "abstained": True,
                    "latency_ms": 0.0,
                    "backend": "error",
                    "details": {"error": str(exc)},
                }
            preds.append(out["prediction"])
            scores.append(float(out.get("score", 0.0)))
            abstained.append(bool(out.get("abstained", False)))
            latencies.append(float(out.get("latency_ms", 0.0)))
            backends.append(str(out.get("backend", "")))
            details_list.append(out.get("details", {}))

        col_pred = f"pred_{method_name}"
        col_score = f"score_{method_name}"
        col_abs = f"abstain_{method_name}"
        col_lat = f"latency_ms_{method_name}"
        results_df[col_pred] = preds
        results_df[col_score] = scores
        results_df[col_abs] = abstained
        results_df[col_lat] = latencies
        results_df[f"backend_{method_name}"] = backends

        # Metrics on non-abstained, non-error examples only
        mask = [
            (p in (LABEL_SUPPORTED, LABEL_HALLUCINATED)) and (not a)
            for p, a in zip(preds, abstained)
        ]
        y_true_eval = [y for y, m in zip(y_true, mask) if m]
        y_pred_eval = [p for p, m in zip(preds, mask) if m]

        metrics = compute_binary_metrics(y_true_eval, y_pred_eval)
        metrics["latency"] = aggregate_latency(latencies)
        n_abstain = sum(1 for a in abstained if a)
        n_error = sum(1 for p in preds if p == -2)
        metrics["abstention_count"] = n_abstain
        metrics["error_count"] = n_error
        metrics["coverage"] = (n - n_abstain - n_error) / n if n else 0.0
        metrics["n_total"] = n
        metrics["backend"] = method_configs[method_name].get("backend", backends[0] if backends else "")

        if "error_type" in results_df.columns:
            eval_df = results_df.loc[mask] if any(mask) else results_df.iloc[0:0]
            if len(eval_df) > 0:
                metrics["error_type_breakdown"] = error_type_breakdown(
                    eval_df, pred_col=col_pred
                ).to_dict(orient="records")
            else:
                metrics["error_type_breakdown"] = []

        all_metrics[method_name] = metrics

    total_ms = (time.perf_counter() - total_start) * 1000.0
    if progress_callback:
        progress_callback(1.0, "Done.")

    return {
        "results_df": results_df,
        "metrics": all_metrics,
        "configs": method_configs,
        "total_time_ms": total_ms,
        "n_examples": n,
        "methods": valid_methods,
    }
