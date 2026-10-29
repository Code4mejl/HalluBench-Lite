"""HalluBench Lite — lightweight hallucination detection benchmarking toolkit."""

__version__ = "0.1.0"

LABEL_SUPPORTED = 0
LABEL_HALLUCINATED = 1

LABEL_NAMES = {
    LABEL_SUPPORTED: "supported",
    LABEL_HALLUCINATED: "hallucinated",
}

POSITIVE_CLASS = LABEL_HALLUCINATED  # positive class for metrics: hallucinated/unsupported
