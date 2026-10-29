"""Hybrid detector combining similarity and NLI signals with abstention."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from hallubench import LABEL_HALLUCINATED, LABEL_SUPPORTED
from detectors.similarity_detector import SimilarityDetector
from detectors.nli_detector import NLIDetector


class HybridDetector:
    """Combine semantic similarity and NLI with documented decision rules.

    Decision rules (configurable)
    -----------------------------
    1. Compute similarity score s and NLI probabilities (e, c, n).
    2. If NLI is unavailable, fall back to similarity-only (no abstention).
    3. If |e - c| < uncertainty_threshold and s is near the decision boundary,
       abstain (insufficient / conflicting evidence).
    4. Else if e >= entailment_threshold and s >= sim_threshold → supported.
    5. Else if c >= contradiction_threshold or s < sim_threshold → hallucinated.
    6. Else → hallucinated (unsupported by evidence).

    Abstentions are reported separately and are NOT counted as correct predictions.
    """

    name = "hybrid"
    display_name = "Hybrid (Similarity + NLI)"
    supports_abstain = True

    def __init__(
        self,
        sim_threshold: float = 0.55,
        entailment_threshold: float = 0.5,
        contradiction_threshold: float = 0.4,
        uncertainty_threshold: float = 0.15,
        force_sim_fallback: bool = False,
    ):
        self.sim_threshold = float(sim_threshold)
        self.entailment_threshold = float(entailment_threshold)
        self.contradiction_threshold = float(contradiction_threshold)
        self.uncertainty_threshold = float(uncertainty_threshold)
        self.force_sim_fallback = force_sim_fallback

        self.sim_detector = SimilarityDetector(
            threshold=sim_threshold, force_fallback=force_sim_fallback
        )
        self.nli_detector = NLIDetector(
            entailment_threshold=entailment_threshold,
            contradiction_threshold=contradiction_threshold,
        )
        self._nli_ok: Optional[bool] = None

    def _ensure_nli(self) -> bool:
        if self._nli_ok is None:
            self._nli_ok = self.nli_detector.is_available()
        return self._nli_ok

    def predict_one(self, evidence: str, claim: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        sim_result = self.sim_detector.predict_one(evidence, claim)
        s = sim_result["score"]

        nli_available = self._ensure_nli()
        if not nli_available:
            # Pure similarity fallback — no abstention
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "prediction": sim_result["prediction"],
                "score": s,
                "abstained": False,
                "latency_ms": latency_ms,
                "backend": f"hybrid_sim_only/{sim_result['backend']}",
                "details": {
                    "similarity": s,
                    "nli_available": False,
                    "rule": "similarity_fallback",
                },
            }

        nli_result = self.nli_detector.predict_one(evidence, claim)
        probs = nli_result["details"]
        e = probs["entailment"]
        c = probs["contradiction"]

        # Uncertainty / conflict check
        gap = abs(e - c)
        near_boundary = abs(s - self.sim_threshold) < 0.1
        if gap < self.uncertainty_threshold and near_boundary:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "prediction": -1,  # sentinel for abstain
                "score": s,
                "abstained": True,
                "latency_ms": latency_ms,
                "backend": "hybrid",
                "details": {
                    "similarity": s,
                    "entailment": e,
                    "contradiction": c,
                    "gap": gap,
                    "rule": "abstain_uncertain",
                },
            }

        if e >= self.entailment_threshold and s >= self.sim_threshold:
            label = LABEL_SUPPORTED
            rule = "entail_and_similar"
        elif c >= self.contradiction_threshold or s < self.sim_threshold:
            label = LABEL_HALLUCINATED
            rule = "contradict_or_dissimilar"
        else:
            label = LABEL_HALLUCINATED
            rule = "default_unsupported"

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "prediction": label,
            "score": e,
            "abstained": False,
            "latency_ms": latency_ms,
            "backend": "hybrid",
            "details": {
                "similarity": s,
                "entailment": e,
                "contradiction": c,
                "gap": gap,
                "rule": rule,
            },
        }

    def predict_batch(
        self, evidences: List[str], claims: List[str]
    ) -> List[Dict[str, Any]]:
        return [self.predict_one(e, c) for e, c in zip(evidences, claims)]

    def get_config(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "sim_threshold": self.sim_threshold,
            "entailment_threshold": self.entailment_threshold,
            "contradiction_threshold": self.contradiction_threshold,
            "uncertainty_threshold": self.uncertainty_threshold,
            "nli_available": self._ensure_nli(),
            "sim_backend": self.sim_detector.backend,
            "nli_backend": self.nli_detector.backend if self._ensure_nli() else "unavailable",
        }
