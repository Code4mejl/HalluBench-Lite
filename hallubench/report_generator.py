"""Generate a Markdown research report from benchmark results."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from hallubench import LABEL_HALLUCINATED, LABEL_SUPPORTED
from hallubench.data_loader import label_distribution
from hallubench.metrics import explain_confusion


def generate_markdown_report(
    results: Dict[str, Any],
    dataset_name: str = "sample_dataset",
    original_df: Optional[pd.DataFrame] = None,
) -> str:
    """Build a complete Markdown research report string."""
    lines: List[str] = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines.append("# HalluBench Lite — Evaluation Report")
    lines.append("")
    lines.append(f"**Generated:** {now}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Dataset")
    lines.append("")
    lines.append(f"- **Name:** `{dataset_name}`")
    n = results.get("n_examples", 0)
    lines.append(f"- **Examples evaluated:** {n}")
    lines.append("")

    if original_df is not None and "label" in original_df.columns:
        dist = label_distribution(original_df)
        lines.append("### Label distribution (full dataset)")
        lines.append("")
        lines.append(f"| Label | Count | Percentage |")
        lines.append(f"|-------|------:|----------:|")
        lines.append(f"| Supported (0) | {dist.get('supported', 0)} | {dist.get('supported_pct', 0)}% |")
        lines.append(f"| Hallucinated (1) | {dist.get('hallucinated', 0)} | {dist.get('hallucinated_pct', 0)}% |")
        lines.append(f"| **Total** | {dist.get('total', 0)} | 100% |")
        lines.append("")
        lines.append(
            "> **Note:** Label `0` = supported / non-hallucinated by the supplied evidence; "
            "`1` = hallucinated / unsupported. An unsupported claim is not necessarily "
            "false in the real world — it may simply lack support in the provided evidence."
        )
        lines.append("")

    lines.append("## 2. Methods and configurations")
    lines.append("")
    configs = results.get("configs", {})
    for method, cfg in configs.items():
        lines.append(f"### {method}")
        lines.append("")
        lines.append("```")
        for k, v in cfg.items():
            lines.append(f"{k}: {v}")
        lines.append("```")
        lines.append("")

    lines.append("## 3. Evaluation metrics")
    lines.append("")
    lines.append(
        "Positive class = hallucinated / unsupported (1). "
        "Abstentions are excluded from accuracy/F1 and reported separately."
    )
    lines.append("")

    metrics = results.get("metrics", {})
    for method, m in metrics.items():
        lines.append(f"### {method}")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|------:|")
        lines.append(f"| Evaluated (non-abstain) | {m.get('n_evaluated', 0)} |")
        lines.append(f"| Coverage | {m.get('coverage', 0):.3f} |")
        lines.append(f"| Abstentions | {m.get('abstention_count', 0)} |")
        lines.append(f"| Errors | {m.get('error_count', 0)} |")
        lines.append(f"| Accuracy | {m.get('accuracy', 0):.4f} |")
        lines.append(f"| Precision (hallucinated) | {m.get('precision', 0):.4f} |")
        lines.append(f"| Recall (hallucinated) | {m.get('recall', 0):.4f} |")
        lines.append(f"| F1 (hallucinated) | {m.get('f1', 0):.4f} |")
        lines.append(f"| Macro F1 | {m.get('macro_f1', 0):.4f} |")
        lat = m.get("latency", {})
        lines.append(f"| Mean latency (ms) | {lat.get('mean_ms', 0):.2f} |")
        lines.append(f"| Total latency (ms) | {lat.get('total_ms', 0):.2f} |")
        lines.append(f"| Backend | {m.get('backend', 'n/a')} |")
        lines.append("")

        cm = m.get("confusion_matrix", [[0, 0], [0, 0]])
        lines.append("#### Confusion matrix (rows=true, cols=predicted)")
        lines.append("")
        lines.append("|  | Pred Supported | Pred Hallucinated |")
        lines.append("|--|----------------:|------------------:|")
        lines.append(f"| **True Supported** | {cm[0][0]} | {cm[0][1]} |")
        lines.append(f"| **True Hallucinated** | {cm[1][0]} | {cm[1][1]} |")
        lines.append("")
        explanations = explain_confusion(cm)
        for _, text in explanations.items():
            lines.append(f"- {text}")
        lines.append("")

    lines.append("## 4. Latency summary")
    lines.append("")
    total_ms = results.get("total_time_ms", 0)
    lines.append(f"- **Wall-clock total:** {total_ms:.1f} ms")
    lines.append("")

    lines.append("## 5. Limitations and reproducibility")
    lines.append("")
    lines.append(
        "- Results depend on the selected dataset, thresholds, and model backends."
    )
    lines.append(
        "- Semantic similarity measures relatedness to the supplied evidence, "
        "**not** absolute factual truth."
    )
    lines.append(
        "- NLI models may be unavailable on constrained hosts; the application falls "
        "back to lexical overlap for the similarity detector and reports the active backend."
    )
    lines.append(
        "- No statistical significance tests were performed. Scores are descriptive "
        "only for the evaluated sample."
    )
    lines.append(
        "- This is an educational research prototype, not a production fact-checking system."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Report generated by HalluBench Lite.*")
    lines.append("")

    return "\n".join(lines)
