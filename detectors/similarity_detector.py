"""Semantic similarity baseline detector with lexical-overlap fallback."""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from hallubench import LABEL_HALLUCINATED, LABEL_SUPPORTED

# Global cache so Streamlit does not reload the model on every rerun
_MODEL_CACHE: Dict[str, Any] = {}


def _tokenize(text: str) -> set:
    """Simple whitespace + punctuation tokenizer for lexical fallback."""
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    return set(tokens)


def lexical_overlap_score(evidence: str, claim: str) -> float:
    """Jaccard similarity between token sets (deterministic fallback)."""
    e_tok = _tokenize(evidence)
    c_tok = _tokenize(claim)
    if not e_tok and not c_tok:
        return 1.0
    if not e_tok or not c_tok:
        return 0.0
    inter = len(e_tok & c_tok)
    union = len(e_tok | c_tok)
    return inter / union if union else 0.0


class SimilarityDetector:
    """Compare evidence and claim via sentence embeddings or lexical overlap.

    Prediction rule
    ---------------
    score = cosine_similarity(embed(evidence), embed(claim))
    If score >= threshold  → supported (0)
    Else                   → hallucinated (1)

    Limitations
    -----------
    High similarity does not prove factual truth; it only measures surface
    semantic relatedness to the supplied evidence.
    """

    name = "semantic_similarity"
    display_name = "Semantic Similarity"
    supports_abstain = False

    def __init__(
        self,
        threshold: float = 0.55,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        force_fallback: bool = False,
    ):
        self.threshold = float(threshold)
        self.model_name = model_name
        self.force_fallback = force_fallback
        self._model = None
        self._backend: str = "uninitialized"
        self._load_error: Optional[str] = None

    def _load_model(self) -> None:
        if self.force_fallback:
            self._backend = "lexical_overlap"
            self._model = None
            return
        cache_key = self.model_name
        if cache_key in _MODEL_CACHE:
            self._model = _MODEL_CACHE[cache_key]
            self._backend = "sentence_transformers"
            return
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(self.model_name)
            _MODEL_CACHE[cache_key] = model
            self._model = model
            self._backend = "sentence_transformers"
        except Exception as exc:
            self._load_error = str(exc)
            self._backend = "lexical_overlap"
            self._model = None

    @property
    def backend(self) -> str:
        if self._backend == "uninitialized":
            self._load_model()
        return self._backend

    def score(self, evidence: str, claim: str) -> float:
        """Return similarity score in [0, 1]."""
        if self._backend == "uninitialized":
            self._load_model()
        if self._backend == "sentence_transformers" and self._model is not None:
            emb = self._model.encode([evidence, claim], convert_to_numpy=True)
            a, b = emb[0], emb[1]
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            if denom < 1e-12:
                return 0.0
            sim = float(np.dot(a, b) / denom)
            # cosine can be slightly outside [-1,1] due to numerics
            return max(0.0, min(1.0, (sim + 1.0) / 2.0)) if sim < 0 else max(0.0, min(1.0, sim))
        return lexical_overlap_score(evidence, claim)

    def predict_one(self, evidence: str, claim: str) -> Dict[str, Any]:
        """Predict a single example. Returns label, score, latency_ms, backend."""
        t0 = time.perf_counter()
        score = self.score(evidence, claim)
        # Higher similarity → more likely supported
        label = LABEL_SUPPORTED if score >= self.threshold else LABEL_HALLUCINATED
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "prediction": label,
            "score": score,
            "abstained": False,
            "latency_ms": latency_ms,
            "backend": self.backend,
            "details": {
                "threshold": self.threshold,
                "similarity": score,
            },
        }

    def predict_batch(
        self, evidences: List[str], claims: List[str]
    ) -> List[Dict[str, Any]]:
        return [self.predict_one(e, c) for e, c in zip(evidences, claims)]

    def get_config(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "threshold": self.threshold,
            "model_name": self.model_name,
            "backend": self.backend,
            "load_error": self._load_error,
            "force_fallback": self.force_fallback,
        }
