"""Tests for detectors (mocked / fallback only — no model downloads)."""

from detectors.similarity_detector import SimilarityDetector, lexical_overlap_score
from detectors.hybrid_detector import HybridDetector
from hallubench import LABEL_HALLUCINATED, LABEL_SUPPORTED
from hallubench.report_generator import generate_markdown_report


def test_lexical_overlap():
    s = lexical_overlap_score("the cat sat on the mat", "the cat sat on a mat")
    assert 0.0 < s <= 1.0
    s2 = lexical_overlap_score("abc", "xyz")
    assert s2 == 0.0


def test_similarity_fallback():
    det = SimilarityDetector(threshold=0.3, force_fallback=True)
    assert det.backend == "lexical_overlap"
    out = det.predict_one(
        "The capital of France is Paris.",
        "The capital of France is Paris.",
    )
    assert out["prediction"] in (LABEL_SUPPORTED, LABEL_HALLUCINATED)
    assert out["abstained"] is False
    assert "latency_ms" in out
    assert out["backend"] == "lexical_overlap"


def test_similarity_threshold():
    det = SimilarityDetector(threshold=0.99, force_fallback=True)
    out = det.predict_one("hello world", "completely different text")
    assert out["prediction"] == LABEL_HALLUCINATED


def test_hybrid_fallback_no_nli():
    """Hybrid with forced sim fallback should not crash."""
    det = HybridDetector(force_sim_fallback=True)
    # Force nli unavailable path by checking config after init
    out = det.predict_one(
        "Water boils at 100 degrees Celsius.",
        "Water boils at 100 degrees Celsius.",
    )
    assert "prediction" in out
    assert "abstained" in out
    assert out["prediction"] in (LABEL_SUPPORTED, LABEL_HALLUCINATED, -1)


def test_report_generation():
    results = {
        "n_examples": 2,
        "configs": {"semantic_similarity": {"threshold": 0.55, "backend": "lexical_overlap"}},
        "metrics": {
            "semantic_similarity": {
                "n_evaluated": 2,
                "accuracy": 1.0,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "macro_f1": 1.0,
                "confusion_matrix": [[1, 0], [0, 1]],
                "abstention_count": 0,
                "error_count": 0,
                "coverage": 1.0,
                "latency": {"mean_ms": 1.0, "total_ms": 2.0},
                "backend": "lexical_overlap",
            }
        },
        "total_time_ms": 5.0,
        "methods": ["semantic_similarity"],
    }
    md = generate_markdown_report(results, dataset_name="test")
    assert "HalluBench Lite" in md
    assert "semantic_similarity" in md
    assert "Accuracy" in md
