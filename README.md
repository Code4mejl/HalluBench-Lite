# HalluBench Lite

**Lightweight research dashboard for benchmarking hallucination-detection methods**

HalluBench Lite evaluates multiple hallucination-detection techniques against labelled factual claims or question–answer pairs. It reports standard metrics, confusion matrices, latency, and example-level explanations — designed as an educational research prototype, **not** a production fact-checking system.

---

## Problem statement

Large language models can produce statements that sound convincing but are factually incorrect or unsupported by available evidence. Developers need a practical way to benchmark hallucination-detection techniques on the same labelled examples and compare reliability, speed, and error patterns.

## Proposed solution

A Streamlit dashboard that:

1. Loads a curated sample dataset (or a user-uploaded CSV).
2. Runs one or more detection methods (semantic similarity, NLI, hybrid).
3. Computes accuracy, precision, recall, F1, macro-F1, confusion matrices, latency, and abstention rates.
4. Provides dataset exploration, error analysis, and a downloadable Markdown research report.

All methods assess claims **against the supplied evidence**, not against an external knowledge base. Semantic similarity is **not** equivalent to factual truth.

---

## Architecture

```
HalluBench-Lite/
├── app.py                      # Streamlit entry point
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/config.toml
├── data/
│   ├── sample_dataset.csv      # 36 curated examples
│   └── dataset_template.csv
├── hallubench/
│   ├── __init__.py             # Label constants
│   ├── data_loader.py
│   ├── validation.py
│   ├── metrics.py
│   ├── benchmark.py
│   ├── models.py               # (reserved)
│   └── report_generator.py
├── detectors/
│   ├── __init__.py
│   ├── similarity_detector.py
│   ├── nli_detector.py
│   └── hybrid_detector.py
└── tests/
    ├── test_data_loader.py
    ├── test_validation.py
    ├── test_metrics.py
    └── test_detectors.py
```

---

## Features

| Page | Description |
|------|-------------|
| **Overview** | Metric cards, comparison charts, dataset summary |
| **Run Benchmark** | Load sample / upload CSV, select methods, progress bar, CSV downloads |
| **Compare Methods** | Documentation of the three detectors and live configs |
| **Dataset Explorer** | Label distribution, filters, TP/TN/FP/FN examples |
| **Error Analysis** | Confusion matrices, classification reports, latency, error-type breakdown |
| **Research Report** | Downloadable Markdown / plain-text report |

---

## Detection methods

### 1. Semantic similarity baseline

- Encode evidence and claim with a lightweight sentence-transformer (`all-MiniLM-L6-v2`) when available.
- Score = cosine similarity; if score ≥ threshold → **supported (0)**, else **hallucinated (1)**.
- Deterministic **lexical Jaccard overlap** fallback when the embedding model cannot load.
- Active backend is always reported.

### 2. NLI-based verification

- Lightweight MNLI model (`typeform/distilbert-base-uncased-mnli` by default).
- High entailment → supported; high contradiction or neutral (configurable) → hallucinated.
- If the model fails to load, a clear error is raised — **predictions are never fabricated**.

### 3. Hybrid detector

Decision rules (configurable):

1. Compute similarity *s* and NLI probabilities (entailment *e*, contradiction *c*).
2. If NLI unavailable → similarity-only (no abstention).
3. If |*e* − *c*| < uncertainty_threshold **and** *s* is near the decision boundary → **abstain**.
4. Else if *e* ≥ entailment_threshold and *s* ≥ sim_threshold → supported.
5. Else if *c* ≥ contradiction_threshold or *s* < sim_threshold → hallucinated.
6. Else → hallucinated (unsupported).

Abstentions are excluded from accuracy/F1 and reported as coverage.

---

## Dataset schema

| Column | Type | Description |
|--------|------|-------------|
| `id` | int | Unique identifier |
| `evidence` | str | Supporting context |
| `claim` | str | Statement to evaluate |
| `label` | int | `0` = supported / non-hallucinated; `1` = hallucinated / unsupported |
| `error_type` | str | `none`, `numerical_error`, `entity_substitution`, `contradiction`, `unsupported`, … |

An unsupported claim is **not** necessarily false in the real world; it may simply lack support in the provided evidence.

The bundled `data/sample_dataset.csv` contains 36 manually curated examples covering supported claims, contradictions, numerical errors, and entity substitutions.

---

## Installation (Windows PowerShell)

```powershell
# 1. Check Python and Git
python --version          # expect 3.11 or 3.12
git --version

# 2. Clone (or download) the repository
git clone https://github.com/<your-username>/HalluBench-Lite.git
cd HalluBench-Lite

# 3. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# If execution policy blocks the script:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. Run tests (no model downloads required)
python -m pytest tests/ -v

# 6. Launch the app
python -m streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

---

## Testing

```powershell
python -m pytest tests/ -v
```

Unit tests cover CSV loading, schema validation, metric edge cases, lexical-overlap detector behaviour, hybrid abstention paths, and report generation. Expensive model calls are not required for the default suite.

---

## Deployment to Streamlit Community Cloud

1. Push the repository to GitHub (public or private).
2. Go to [https://share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app** → select the repository, branch (`main`), and main file path `app.py`.
4. Click **Deploy**.

### Notes

- Cold starts may take 30–90 seconds while dependencies install.
- The NLI / embedding models will download on first use and may exceed free-tier memory on large samples. Use the **Force lexical-overlap fallback** option or a small sample limit for reliable free-tier runs.
- Do not commit secrets, model weights, or virtual environments (see `.gitignore`).

---

## Screenshots

> Place genuine screenshots here after running the app locally:
>
> - `docs/overview.png` — Overview page with metric cards
> - `docs/run_benchmark.png` — Benchmark progress and results table
> - `docs/error_analysis.png` — Confusion matrix and metrics

---

## Research references

This project is an **independent lightweight comparison**, not an official reproduction of any paper.

1. **HaluEval** — A Large-Scale Hallucination Evaluation Benchmark for Large Language Models.  
   Junyi Li et al., EMNLP 2023.  
   https://aclanthology.org/2023.emnlp-main.397/  
   *Inspiration for labelled claim/evidence evaluation design.*

2. **SelfCheckGPT** — Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models.  
   Potsawee Manakul et al., EMNLP 2023.  
   https://aclanthology.org/2023.emnlp-main.557/  
   *Inspiration for consistency-based detection ideas (not implemented here).*

3. **RAGTruth** — A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models.  
   Yuanhao Wu et al., ACL 2024.  
   https://aclanthology.org/2024.acl-long.585/  
   *Inspiration for evidence-grounded label schemas.*

4. **Enhancing Uncertainty-Based Hallucination Detection with Stronger Focus.**  
   Tianyu Liu et al., EMNLP 2023.  
   https://aclanthology.org/2023.emnlp-main.58/  
   *Inspiration for uncertainty / abstention handling in the hybrid detector.*

5. **A Hybrid Framework for Detecting Hallucinations in Large Language Model Outputs.**  
   IEEE ICAAI C 2025.  
   https://doi.org/10.1109/ICAAIC64647.2025.11330957  
   *Inspiration for combining multiple signals; implementation here is independent and simplified.*

---

## Limitations

- Results depend on the selected dataset, thresholds, and model backends.
- Semantic similarity measures relatedness to the supplied evidence only.
- NLI models may be unavailable on constrained hosts; the UI reports the active backend.
- No statistical significance tests are performed.
- Sample dataset is small (36 examples) and for demonstration only — not a published research benchmark.
- Free hosting may time out or run out of memory when downloading transformer models.

## Future improvements

- Additional detectors (SelfCheckGPT-style sampling, token-level uncertainty).
- Larger curated datasets with richer error-type annotations.
- Optional GPU acceleration and batch encoding.
- Export of interactive Plotly charts.
- Threshold tuning UI with hold-out validation.

---

## License

Educational / research prototype. Use at your own discretion.
