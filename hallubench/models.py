"""Reserved module for shared model utilities and caching helpers.

Model loading is currently implemented inside each detector
(similarity_detector, nli_detector) with module-level caches so that
Streamlit reruns do not re-download weights.
"""

from __future__ import annotations

# Intentionally minimal — detectors own their loaders.
