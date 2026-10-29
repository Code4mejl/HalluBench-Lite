"""Tests for CSV validation."""

import pandas as pd
import pytest

from hallubench.validation import validate_and_clean, validate_labels, validate_schema


def test_valid_schema():
    df = pd.DataFrame(
        {
            "id": [1],
            "evidence": ["e"],
            "claim": ["c"],
            "label": [0],
        }
    )
    ok, errs = validate_schema(df)
    assert ok
    assert errs == []


def test_missing_columns():
    df = pd.DataFrame({"id": [1], "claim": ["c"]})
    ok, errs = validate_schema(df)
    assert not ok
    assert any("Missing" in e for e in errs)


def test_empty_dataset():
    df = pd.DataFrame(columns=["id", "evidence", "claim", "label"])
    ok, errs = validate_schema(df)
    assert not ok


def test_invalid_labels():
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "evidence": ["a", "b"],
            "claim": ["c", "d"],
            "label": [0, 5],
        }
    )
    ok, errs = validate_labels(df)
    assert not ok


def test_validate_and_clean():
    df = pd.DataFrame(
        {
            "id": [1],
            "evidence": ["e"],
            "claim": ["c"],
            "label": [1],
        }
    )
    clean, warnings = validate_and_clean(df)
    assert warnings == []
    assert clean["label"].iloc[0] == 1
    assert "error_type" in clean.columns
