"""NLI-based hallucination detector with graceful fallback."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from hallubench import LABEL_HALLUCINATED, LABEL_SUPPORTED

_MODEL_CACHE: Dict[str, Any] = {}
_TOKENIZER_CACHE: Dict[str, Any] = {}


class NLIDetector:
    """Use a lightweight NLI model to check whether evidence entails the claim.

    Label mapping (configurable)
    ----------------------------
    - entailment probability high  → supported (0)
    - contradiction probability high → hallucinated (1)
    - neutral / low confidence → treated as hallucinated (unsupported) by default,
      because the claim is not supported by the evidence.

    The default model is ``cross-encoder/nli-deberta-v3-xsmall`` or
    ``typeform/distilbert-base-uncased-mnli`` depending on availability.
    If no model can be loaded, the detector raises a clear error rather than
    inventing predictions.
    """

    name = "nli"
    display_name = "NLI Verification"
    supports_abstain = False

    # Label indices for standard MNLI: 0=contradiction, 1=entailment, 2=neutral
    # (some models use different order; we detect from config when possible)
    def __init__(
        self,
        entailment_threshold: float = 0.5,
        contradiction_threshold: float = 0.4,
        model_name: str = "typeform/distilbert-base-uncased-mnli",
        treat_neutral_as_hallucinated: bool = True,
    ):
        self.entailment_threshold = float(entailment_threshold)
        self.contradiction_threshold = float(contradiction_threshold)
        self.model_name = model_name
        self.treat_neutral_as_hallucinated = treat_neutral_as_hallucinated
        self._model = None
        self._tokenizer = None
        self._backend = "uninitialized"
        self._load_error: Optional[str] = None
        self._label2id: Optional[Dict[str, int]] = None

    def _load_model(self) -> None:
        if self._backend != "uninitialized":
            return
        cache_key = self.model_name
        if cache_key in _MODEL_CACHE and cache_key in _TOKENIZER_CACHE:
            self._model = _MODEL_CACHE[cache_key]
            self._tokenizer = _TOKENIZER_CACHE[cache_key]
            self._backend = "transformers"
            self._label2id = getattr(self._model.config, "label2id", None)
            return
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch

            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            model.eval()
            _TOKENIZER_CACHE[cache_key] = tokenizer
            _MODEL_CACHE[cache_key] = model
            self._tokenizer = tokenizer
            self._model = model
            self._backend = "transformers"
            self._label2id = getattr(model.config, "label2id", None)
        except Exception as exc:
            self._load_error = str(exc)
            self._backend = "unavailable"
            self._model = None
            self._tokenizer = None

    @property
    def backend(self) -> str:
        if self._backend == "uninitialized":
            self._load_model()
        return self._backend

    def is_available(self) -> bool:
        return self.backend == "transformers"

    def _nli_probs(self, evidence: str, claim: str) -> Dict[str, float]:
        """Return probabilities for entailment, contradiction, neutral."""
        import torch

        inputs = self._tokenizer(
            evidence,
            claim,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        )
        with torch.no_grad():
            logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze().tolist()

        # Map indices to names
        id2label = getattr(self._model.config, "id2label", None)
        if id2label is None and self._label2id:
            id2label = {v: k for k, v in self._label2id.items()}

        result = {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}
        if id2label:
            for idx, p in enumerate(probs):
                name = str(id2label.get(idx, "")).lower()
                if "entail" in name:
                    result["entailment"] = float(p)
                elif "contradict" in name:
                    result["contradiction"] = float(p)
                elif "neutral" in name:
                    result["neutral"] = float(p)
        else:
            # Common MNLI order used by many models: contradiction, entailment, neutral
            if len(probs) >= 3:
                result["contradiction"] = float(probs[0])
                result["entailment"] = float(probs[1])
                result["neutral"] = float(probs[2])
            elif len(probs) == 2:
                result["contradiction"] = float(probs[0])
                result["entailment"] = float(probs[1])
        return result

    def predict_one(self, evidence: str, claim: str) -> Dict[str, Any]:
        if self.backend == "unavailable":
            raise RuntimeError(
                f"NLI model could not be loaded: {self._load_error}. "
                "Use semantic_similarity or hybrid (with fallback) instead, "
                "or install transformers and torch."
            )
        t0 = time.perf_counter()
        probs = self._nli_probs(evidence, claim)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if probs["entailment"] >= self.entailment_threshold:
            label = LABEL_SUPPORTED
        elif probs["contradiction"] >= self.contradiction_threshold:
            label = LABEL_HALLUCINATED
        elif self.treat_neutral_as_hallucinated:
            label = LABEL_HALLUCINATED
        else:
            label = LABEL_SUPPORTED

        return {
            "prediction": label,
            "score": probs["entailment"],
            "abstained": False,
            "latency_ms": latency_ms,
            "backend": self.backend,
            "details": {
                "entailment": probs["entailment"],
                "contradiction": probs["contradiction"],
                "neutral": probs["neutral"],
                "entailment_threshold": self.entailment_threshold,
                "contradiction_threshold": self.contradiction_threshold,
            },
        }

    def predict_batch(
        self, evidences: List[str], claims: List[str]
    ) -> List[Dict[str, Any]]:
        return [self.predict_one(e, c) for e, c in zip(evidences, claims)]

    def get_config(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model_name": self.model_name,
            "entailment_threshold": self.entailment_threshold,
            "contradiction_threshold": self.contradiction_threshold,
            "treat_neutral_as_hallucinated": self.treat_neutral_as_hallucinated,
            "backend": self.backend,
            "load_error": self._load_error,
        }
