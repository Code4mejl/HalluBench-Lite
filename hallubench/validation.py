"""CSV schema and label validation."""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd

from hallubench import LABEL_HALLUCINATED, LABEL_SUPPORTED
from hallubench.data_loader import REQUIRED_COLUMNS


def validate_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Check that required columns exist.

    Returns
    -------
    ok : bool
    messages : list of error strings (empty if ok)
    """
    errors: List[str] = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {', '.join(missing)}")
    if df.empty:
        errors.append("Dataset contains no rows.")
    return len(errors) == 0, errors


def validate_labels(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Ensure label column contains only 0 and 1."""
    errors: List[str] = []
    if "label" not in df.columns:
        errors.append("Column 'label' is missing.")
        return False, errors
    try:
        labels = pd.to_numeric(df["label"], errors="coerce")
    except Exception:
        errors.append("Could not parse 'label' column as numbers.")
        return False, errors
    if labels.isna().any():
        bad = df.loc[labels.isna(), "id"].tolist() if "id" in df.columns else []
        errors.append(f"Non-numeric or missing labels found (ids: {bad[:5]}{'...' if len(bad) > 5 else ''}).")
    unique = set(labels.dropna().unique().tolist())
    allowed = {LABEL_SUPPORTED, LABEL_HALLUCINATED, float(LABEL_SUPPORTED), float(LABEL_HALLUCINATED)}
    invalid = unique - allowed
    if invalid:
        errors.append(f"Invalid label values: {sorted(invalid)}. Expected only 0 (supported) or 1 (hallucinated).")
    return len(errors) == 0, errors


def validate_and_clean(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Validate schema and labels; coerce label to int; return cleaned copy and warnings."""
    ok_schema, schema_errs = validate_schema(df)
    if not ok_schema:
        return df, schema_errs

    ok_labels, label_errs = validate_labels(df)
    warnings = schema_errs + label_errs
    if not ok_labels:
        return df, warnings

    clean = df.copy()
    clean["label"] = pd.to_numeric(clean["label"], errors="coerce").astype(int)
    # Ensure string columns are strings
    for col in ("evidence", "claim"):
        if col in clean.columns:
            clean[col] = clean[col].astype(str).fillna("")
    if "error_type" not in clean.columns:
        clean["error_type"] = "none"
    else:
        clean["error_type"] = clean["error_type"].astype(str).fillna("none")
    if "id" not in clean.columns:
        clean["id"] = range(1, len(clean) + 1)
    return clean, warnings


def check_empty_or_invalid(df: pd.DataFrame) -> List[str]:
    """Convenience: return all validation errors for an uploaded frame."""
    _, errs = validate_and_clean(df)
    return errs
