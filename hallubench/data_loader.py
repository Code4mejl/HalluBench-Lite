"""Dataset loading and basic inspection utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from hallubench import LABEL_HALLUCINATED, LABEL_SUPPORTED

REQUIRED_COLUMNS = ["id", "evidence", "claim", "label"]
OPTIONAL_COLUMNS = ["error_type"]

SAMPLE_DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_dataset.csv"
TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "data" / "dataset_template.csv"


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file into a DataFrame.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    ValueError
        If the file is empty or cannot be parsed as CSV.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Failed to parse CSV: {exc}") from exc
    if df.empty:
        raise ValueError("Dataset is empty.")
    return df


def load_sample_dataset() -> pd.DataFrame:
    """Load the bundled sample dataset."""
    return load_csv(SAMPLE_DATASET_PATH)


def get_template_path() -> Path:
    """Return path to the CSV upload template."""
    return TEMPLATE_PATH


def label_distribution(df: pd.DataFrame) -> dict:
    """Return counts and proportions for ground-truth labels."""
    if "label" not in df.columns:
        return {}
    counts = df["label"].value_counts().to_dict()
    total = len(df)
    return {
        "supported": int(counts.get(LABEL_SUPPORTED, 0)),
        "hallucinated": int(counts.get(LABEL_HALLUCINATED, 0)),
        "total": total,
        "supported_pct": round(100.0 * counts.get(LABEL_SUPPORTED, 0) / total, 1) if total else 0.0,
        "hallucinated_pct": round(100.0 * counts.get(LABEL_HALLUCINATED, 0) / total, 1) if total else 0.0,
    }


def preview(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return a head preview of the dataset."""
    return df.head(n).copy()


def filter_dataset(
    df: pd.DataFrame,
    ground_truth: Optional[int] = None,
    prediction: Optional[int] = None,
    error_type: Optional[str] = None,
    method_col: Optional[str] = None,
) -> pd.DataFrame:
    """Filter a results DataFrame by ground-truth, prediction, or error type."""
    out = df.copy()
    if ground_truth is not None and "label" in out.columns:
        out = out[out["label"] == ground_truth]
    if prediction is not None and method_col and method_col in out.columns:
        out = out[out[method_col] == prediction]
    if error_type is not None and "error_type" in out.columns:
        out = out[out["error_type"] == error_type]
    return out.reset_index(drop=True)
