"""Tests for data loading."""

import pandas as pd
import pytest

from hallubench.data_loader import (
    label_distribution,
    load_csv,
    load_sample_dataset,
    preview,
)


def test_load_sample_dataset():
    df = load_sample_dataset()
    assert len(df) >= 30
    assert "evidence" in df.columns
    assert "claim" in df.columns
    assert "label" in df.columns


def test_label_distribution():
    df = load_sample_dataset()
    dist = label_distribution(df)
    assert dist["total"] == len(df)
    assert dist["supported"] + dist["hallucinated"] == dist["total"]


def test_preview():
    df = load_sample_dataset()
    p = preview(df, n=5)
    assert len(p) == 5


def test_load_missing_file():
    with pytest.raises(FileNotFoundError):
        load_csv("/nonexistent/path/dataset.csv")
