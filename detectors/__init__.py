"""Hallucination detectors for HalluBench Lite."""

from detectors.similarity_detector import SimilarityDetector
from detectors.nli_detector import NLIDetector
from detectors.hybrid_detector import HybridDetector

AVAILABLE_METHODS = {
    "semantic_similarity": SimilarityDetector,
    "nli": NLIDetector,
    "hybrid": HybridDetector,
}

__all__ = [
    "SimilarityDetector",
    "NLIDetector",
    "HybridDetector",
    "AVAILABLE_METHODS",
]
