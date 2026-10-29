"""
HalluBench Lite — Streamlit research dashboard for hallucination detection benchmarking.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from hallubench import LABEL_HALLUCINATED, LABEL_NAMES, LABEL_SUPPORTED
from hallubench.benchmark import run_benchmark
from hallubench.data_loader import (
    get_template_path,
    label_distribution,
    load_sample_dataset,
    preview,
)
from hallubench.metrics import explain_confusion
from hallubench.report_generator import generate_markdown_report
from hallubench.validation import validate_and_clean
from detectors import AVAILABLE_METHODS

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="HalluBench Lite",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def _init_state() -> None:
    defaults = {
        "dataset": None,
        "dataset_name": "sample_dataset",
        "results": None,
        "last_error": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("🔬 HalluBench Lite")
st.sidebar.caption("Hallucination detection benchmarking")
page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Run Benchmark",
        "Compare Methods",
        "Dataset Explorer",
        "Error Analysis",
        "Research Report",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Disclaimer:** This tool assists research and does **not** establish "
    "factual truth automatically. Semantic similarity ≠ factual correctness."
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
METHOD_DISPLAY = {
    "semantic_similarity": "Semantic Similarity",
    "nli": "NLI Verification",
    "hybrid": "Hybrid (Sim + NLI)",
}


def metric_card(label: str, value: str, help_text: str = "") -> None:
    st.metric(label, value, help=help_text if help_text else None)


def load_default_dataset() -> None:
    try:
        df = load_sample_dataset()
        clean, warns = validate_and_clean(df)
        st.session_state.dataset = clean
        st.session_state.dataset_name = "sample_dataset"
        st.session_state.last_error = None
        if warns:
            st.warning("; ".join(warns))
    except Exception as exc:
        st.session_state.last_error = str(exc)
        st.error(f"Failed to load sample dataset: {exc}")


# ---------------------------------------------------------------------------
# OVERVIEW
# ---------------------------------------------------------------------------
if page == "Overview":
    st.title("HalluBench Lite")
    st.markdown(
        "A lightweight research dashboard for **benchmarking hallucination-detection "
        "methods** on labelled factual claims. Compare reliability, latency, and "
        "error patterns of semantic similarity, NLI, and hybrid detectors."
    )
    st.info(
        "⚠️ **Important:** Semantic similarity measures relatedness to the supplied "
        "evidence — it is **not** equivalent to factual truth in the real world. "
        "An unsupported claim may simply lack support in the provided context."
    )

    if st.session_state.dataset is None:
        load_default_dataset()

    df = st.session_state.dataset
    results = st.session_state.results

    c1, c2, c3, c4, c5 = st.columns(5)
    n_examples = len(df) if df is not None else 0
    with c1:
        metric_card("Examples", str(n_examples))
    if results and results.get("metrics"):
        methods = results.get("methods", [])
        with c2:
            metric_card("Methods", str(len(methods)))
        # Average accuracy / macro F1 / latency across methods
        accs = [m.get("accuracy", 0) for m in results["metrics"].values()]
        f1s = [m.get("macro_f1", 0) for m in results["metrics"].values()]
        lats = [
            m.get("latency", {}).get("mean_ms", 0)
            for m in results["metrics"].values()
        ]
        with c3:
            metric_card("Avg Accuracy", f"{sum(accs)/len(accs):.3f}" if accs else "—")
        with c4:
            metric_card("Avg Macro F1", f"{sum(f1s)/len(f1s):.3f}" if f1s else "—")
        with c5:
            metric_card("Avg Latency (ms)", f"{sum(lats)/len(lats):.1f}" if lats else "—")
    else:
        with c2:
            metric_card("Methods", "—")
        with c3:
            metric_card("Avg Accuracy", "—")
        with c4:
            metric_card("Avg Macro F1", "—")
        with c5:
            metric_card("Avg Latency (ms)", "—")
        st.caption("Run a benchmark to populate performance cards.")

    st.markdown("---")
    st.subheader("Dataset configuration")
    if df is not None:
        dist = label_distribution(df)
        st.markdown(
            f"- **Dataset:** `{st.session_state.dataset_name}`  \n"
            f"- **Size:** {dist['total']} examples  \n"
            f"- **Supported (0):** {dist['supported']} ({dist['supported_pct']}%)  \n"
            f"- **Hallucinated (1):** {dist['hallucinated']} ({dist['hallucinated_pct']}%)"
        )
    else:
        st.warning("No dataset loaded.")

    if results and results.get("metrics"):
        st.subheader("Method comparison")
        rows = []
        for name, m in results["metrics"].items():
            rows.append(
                {
                    "Method": METHOD_DISPLAY.get(name, name),
                    "Accuracy": m.get("accuracy", 0),
                    "Macro F1": m.get("macro_f1", 0),
                    "F1 (hallucinated)": m.get("f1", 0),
                    "Mean latency (ms)": m.get("latency", {}).get("mean_ms", 0),
                    "Coverage": m.get("coverage", 1.0),
                }
            )
        chart_df = pd.DataFrame(rows)
        fig = px.bar(
            chart_df,
            x="Method",
            y=["Accuracy", "Macro F1", "F1 (hallucinated)"],
            barmode="group",
            title="Performance metrics by method",
            color_discrete_sequence=["#5B5FC7", "#7C83FD", "#A5B4FC"],
        )
        fig.update_layout(yaxis_title="Score", legend_title="")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.bar(
            chart_df,
            x="Method",
            y="Mean latency (ms)",
            title="Average inference latency",
            color="Method",
            color_discrete_sequence=["#5B5FC7", "#7C83FD", "#A5B4FC"],
        )
        st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------------------
# RUN BENCHMARK
# ---------------------------------------------------------------------------
elif page == "Run Benchmark":
    st.title("Run Benchmark")
    st.markdown("Load a dataset, select detection methods, and evaluate.")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Load sample dataset", type="primary"):
            load_default_dataset()
            st.success(f"Loaded sample dataset ({len(st.session_state.dataset)} rows).")
    with col_b:
        template_path = get_template_path()
        if template_path.exists():
            with open(template_path, "rb") as f:
                st.download_button(
                    "Download CSV template",
                    f,
                    file_name="dataset_template.csv",
                    mime="text/csv",
                )

    uploaded = st.file_uploader("Or upload a CSV dataset", type=["csv"])
    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded)
            clean, warns = validate_and_clean(raw)
            if warns and any("Missing" in w or "Invalid" in w or "empty" in w.lower() for w in warns):
                st.error("Validation failed: " + "; ".join(warns))
            else:
                st.session_state.dataset = clean
                st.session_state.dataset_name = uploaded.name
                st.session_state.results = None
                if warns:
                    st.warning("; ".join(warns))
                else:
                    st.success(f"Loaded `{uploaded.name}` ({len(clean)} rows).")
        except Exception as exc:
            st.error(f"Could not parse CSV: {exc}")

    df = st.session_state.dataset
    if df is None:
        st.warning("No dataset loaded. Load the sample or upload a CSV.")
        st.stop()

    st.markdown(f"**Current dataset:** `{st.session_state.dataset_name}` — {len(df)} rows")

    method_options = list(AVAILABLE_METHODS.keys())
    selected = st.multiselect(
        "Detection methods",
        options=method_options,
        default=["semantic_similarity"],
        format_func=lambda x: METHOD_DISPLAY.get(x, x),
    )
    sample_limit = st.number_input(
        "Sample limit (0 = all)",
        min_value=0,
        max_value=len(df),
        value=min(20, len(df)),
        help="Limit examples for faster testing. Set to 0 to use the full dataset.",
    )
    if sample_limit == 0:
        sample_limit = None

    with st.expander("Advanced thresholds"):
        sim_thresh = st.slider("Similarity threshold", 0.0, 1.0, 0.55, 0.05)
        entail_thresh = st.slider("NLI entailment threshold", 0.0, 1.0, 0.50, 0.05)
        contrad_thresh = st.slider("NLI contradiction threshold", 0.0, 1.0, 0.40, 0.05)
        unc_thresh = st.slider("Hybrid uncertainty threshold", 0.0, 0.5, 0.15, 0.05)
        force_fallback = st.checkbox(
            "Force lexical-overlap fallback (no embedding model download)",
            value=False,
        )

    detector_kwargs = {
        "semantic_similarity": {
            "threshold": sim_thresh,
            "force_fallback": force_fallback,
        },
        "nli": {
            "entailment_threshold": entail_thresh,
            "contradiction_threshold": contrad_thresh,
        },
        "hybrid": {
            "sim_threshold": sim_thresh,
            "entailment_threshold": entail_thresh,
            "contradiction_threshold": contrad_thresh,
            "uncertainty_threshold": unc_thresh,
            "force_sim_fallback": force_fallback,
        },
    }

    if st.button("Start benchmark", type="primary", disabled=not selected):
        progress = st.progress(0.0, text="Starting...")
        status = st.empty()

        def cb(frac: float, msg: str) -> None:
            progress.progress(min(frac, 1.0), text=msg)
            status.caption(msg)

        with st.spinner("Running evaluation..."):
            try:
                results = run_benchmark(
                    df,
                    selected,
                    sample_limit=sample_limit,
                    progress_callback=cb,
                    detector_kwargs=detector_kwargs,
                )
                if results.get("error"):
                    st.error(results["error"])
                else:
                    st.session_state.results = results
                    st.success(
                        f"Benchmark complete — {results['n_examples']} examples, "
                        f"{results['total_time_ms']:.0f} ms total."
                    )
            except Exception as exc:
                st.error(f"Benchmark failed: {exc}")
                st.session_state.last_error = str(exc)

    results = st.session_state.results
    if results and results.get("metrics"):
        st.subheader("Results summary")
        summary_rows = []
        for name, m in results["metrics"].items():
            summary_rows.append(
                {
                    "Method": METHOD_DISPLAY.get(name, name),
                    "Accuracy": round(m.get("accuracy", 0), 4),
                    "Precision": round(m.get("precision", 0), 4),
                    "Recall": round(m.get("recall", 0), 4),
                    "F1": round(m.get("f1", 0), 4),
                    "Macro F1": round(m.get("macro_f1", 0), 4),
                    "Coverage": round(m.get("coverage", 0), 3),
                    "Abstentions": m.get("abstention_count", 0),
                    "Mean latency (ms)": round(m.get("latency", {}).get("mean_ms", 0), 2),
                    "Backend": m.get("backend", ""),
                }
            )
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

        # Downloads
        res_df = results["results_df"]
        csv_buf = io.StringIO()
        res_df.to_csv(csv_buf, index=False)
        st.download_button(
            "Download evaluated examples (CSV)",
            csv_buf.getvalue(),
            file_name="hallubench_results.csv",
            mime="text/csv",
        )
        metrics_rows = []
        for name, m in results["metrics"].items():
            metrics_rows.append(
                {
                    "method": name,
                    "accuracy": m.get("accuracy"),
                    "precision": m.get("precision"),
                    "recall": m.get("recall"),
                    "f1": m.get("f1"),
                    "macro_f1": m.get("macro_f1"),
                    "coverage": m.get("coverage"),
                    "abstentions": m.get("abstention_count"),
                    "mean_latency_ms": m.get("latency", {}).get("mean_ms"),
                }
            )
        metrics_csv = pd.DataFrame(metrics_rows).to_csv(index=False)
        st.download_button(
            "Download metrics summary (CSV)",
            metrics_csv,
            file_name="hallubench_metrics.csv",
            mime="text/csv",
        )

# ---------------------------------------------------------------------------
# COMPARE METHODS
# ---------------------------------------------------------------------------
elif page == "Compare Methods":
    st.title("Compare Detection Methods")
    st.markdown(
        "Three reusable detectors. All use the **supplied evidence** as the "
        "reference; none claim absolute real-world truth."
    )

    st.subheader("1. Semantic Similarity Baseline")
    st.markdown(
        """
        - Embeds evidence and claim (or uses lexical Jaccard overlap as fallback).
        - Cosine similarity ≥ threshold → **supported**; otherwise **hallucinated**.
        - Model: `all-MiniLM-L6-v2` when available; otherwise deterministic token overlap.
        - **Limitation:** High similarity ≠ factual truth.
        """
    )

    st.subheader("2. NLI-Based Verification")
    st.markdown(
        """
        - Lightweight MNLI model scores entailment / contradiction / neutral.
        - High entailment → supported; high contradiction or neutral (default) → hallucinated.
        - If the model cannot load, an actionable error is shown — predictions are **not** fabricated.
        - Default model: `typeform/distilbert-base-uncased-mnli`.
        """
    )

    st.subheader("3. Hybrid Detector")
    st.markdown(
        """
        - Combines similarity score and NLI probabilities.
        - **Abstains** when signals conflict and similarity is near the decision boundary.
        - Configurable uncertainty threshold.
        - Falls back to similarity-only when NLI is unavailable.
        - Abstentions are reported separately and **not** counted as correct.
        """
    )

    results = st.session_state.results
    if results and results.get("configs"):
        st.subheader("Active configurations (last run)")
        for name, cfg in results["configs"].items():
            with st.expander(METHOD_DISPLAY.get(name, name)):
                st.json(cfg)
    else:
        st.info("Run a benchmark to see live configurations and backends.")

# ---------------------------------------------------------------------------
# DATASET EXPLORER
# ---------------------------------------------------------------------------
elif page == "Dataset Explorer":
    st.title("Dataset Explorer")
    df = st.session_state.dataset
    if df is None:
        load_default_dataset()
        df = st.session_state.dataset
    if df is None:
        st.warning("No dataset available.")
        st.stop()

    dist = label_distribution(df)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total", dist["total"])
    c2.metric("Supported", f"{dist['supported']} ({dist['supported_pct']}%)")
    c3.metric("Hallucinated", f"{dist['hallucinated']} ({dist['hallucinated_pct']}%)")

    fig = px.pie(
        names=["Supported", "Hallucinated"],
        values=[dist["supported"], dist["hallucinated"]],
        title="Label distribution",
        color_discrete_sequence=["#5B5FC7", "#F87171"],
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Preview")
    st.dataframe(preview(df, 15), use_container_width=True)

    results = st.session_state.results
    if results and "results_df" in results:
        rdf = results["results_df"]
        st.subheader("Filter evaluated examples")
        methods = results.get("methods", [])
        method = st.selectbox(
            "Method", methods, format_func=lambda x: METHOD_DISPLAY.get(x, x)
        )
        pred_col = f"pred_{method}"
        gt_filter = st.selectbox(
            "Ground-truth label",
            ["All", "Supported (0)", "Hallucinated (1)"],
        )
        pred_filter = st.selectbox(
            "Predicted label",
            ["All", "Supported (0)", "Hallucinated (1)", "Abstained"],
        )
        error_types = ["All"] + sorted(rdf["error_type"].dropna().unique().tolist()) if "error_type" in rdf.columns else ["All"]
        et_filter = st.selectbox("Error type", error_types)

        filtered = rdf.copy()
        if gt_filter == "Supported (0)":
            filtered = filtered[filtered["label"] == 0]
        elif gt_filter == "Hallucinated (1)":
            filtered = filtered[filtered["label"] == 1]
        if pred_col in filtered.columns:
            if pred_filter == "Supported (0)":
                filtered = filtered[filtered[pred_col] == 0]
            elif pred_filter == "Hallucinated (1)":
                filtered = filtered[filtered[pred_col] == 1]
            elif pred_filter == "Abstained":
                abs_col = f"abstain_{method}"
                if abs_col in filtered.columns:
                    filtered = filtered[filtered[abs_col] == True]  # noqa: E712

        if et_filter != "All" and "error_type" in filtered.columns:
            filtered = filtered[filtered["error_type"] == et_filter]

        show_cols = ["id", "evidence", "claim", "label"]
        if pred_col in filtered.columns:
            show_cols.append(pred_col)
        score_col = f"score_{method}"
        if score_col in filtered.columns:
            show_cols.append(score_col)
        if "error_type" in filtered.columns:
            show_cols.append("error_type")
        st.dataframe(filtered[show_cols].head(50), use_container_width=True)
        st.caption(f"Showing {min(50, len(filtered))} of {len(filtered)} filtered rows.")

        # TP / TN / FP / FN examples
        if pred_col in rdf.columns:
            st.subheader("Example error categories")
            tabs = st.tabs(["True Positives", "True Negatives", "False Positives", "False Negatives", "Abstentions"])
            with tabs[0]:
                tp = rdf[(rdf["label"] == 1) & (rdf[pred_col] == 1)]
                st.dataframe(tp[show_cols].head(5) if len(tp) else pd.DataFrame(), use_container_width=True)
            with tabs[1]:
                tn = rdf[(rdf["label"] == 0) & (rdf[pred_col] == 0)]
                st.dataframe(tn[show_cols].head(5) if len(tn) else pd.DataFrame(), use_container_width=True)
            with tabs[2]:
                fp = rdf[(rdf["label"] == 0) & (rdf[pred_col] == 1)]
                st.dataframe(fp[show_cols].head(5) if len(fp) else pd.DataFrame(), use_container_width=True)
            with tabs[3]:
                fn = rdf[(rdf["label"] == 1) & (rdf[pred_col] == 0)]
                st.dataframe(fn[show_cols].head(5) if len(fn) else pd.DataFrame(), use_container_width=True)
            with tabs[4]:
                abs_col = f"abstain_{method}"
                if abs_col in rdf.columns:
                    ab = rdf[rdf[abs_col] == True]  # noqa: E712
                    st.dataframe(ab[show_cols].head(5) if len(ab) else pd.DataFrame(), use_container_width=True)
                else:
                    st.info("This method does not abstain.")
    else:
        st.info("Run a benchmark to enable prediction filters and error-category examples.")

# ---------------------------------------------------------------------------
# ERROR ANALYSIS
# ---------------------------------------------------------------------------
elif page == "Error Analysis":
    st.title("Error Analysis")
    results = st.session_state.results
    if not results or not results.get("metrics"):
        st.warning("Run a benchmark first to see error analysis.")
        st.stop()

    for name, m in results["metrics"].items():
        st.subheader(METHOD_DISPLAY.get(name, name))
        cm = m.get("confusion_matrix", [[0, 0], [0, 0]])
        explanations = explain_confusion(cm)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("**Confusion matrix**")
            cm_df = pd.DataFrame(
                cm,
                index=["True Supported", "True Hallucinated"],
                columns=["Pred Supported", "Pred Hallucinated"],
            )
            st.dataframe(cm_df, use_container_width=True)
            for text in explanations.values():
                st.caption(text)
        with col2:
            fig = go.Figure(
                data=go.Heatmap(
                    z=cm,
                    x=["Pred Supported", "Pred Hallucinated"],
                    y=["True Supported", "True Hallucinated"],
                    colorscale="Blues",
                    text=cm,
                    texttemplate="%{text}",
                    showscale=False,
                )
            )
            fig.update_layout(title="Confusion matrix heatmap", height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{m.get('accuracy', 0):.3f}")
        c2.metric("Precision", f"{m.get('precision', 0):.3f}")
        c3.metric("Recall", f"{m.get('recall', 0):.3f}")
        c4.metric("F1", f"{m.get('f1', 0):.3f}")
        c5.metric("Macro F1", f"{m.get('macro_f1', 0):.3f}")

        c6, c7, c8 = st.columns(3)
        c6.metric("Coverage", f"{m.get('coverage', 0):.3f}")
        c7.metric("Abstentions", str(m.get("abstention_count", 0)))
        c8.metric("Mean latency (ms)", f"{m.get('latency', {}).get('mean_ms', 0):.1f}")

        report = m.get("classification_report", {})
        if report:
            with st.expander("Classification report"):
                st.json(report)

        etb = m.get("error_type_breakdown")
        if etb:
            st.markdown("**Error-type breakdown** (from dataset annotations)")
            st.dataframe(pd.DataFrame(etb), use_container_width=True)
        st.markdown("---")

# ---------------------------------------------------------------------------
# RESEARCH REPORT
# ---------------------------------------------------------------------------
elif page == "Research Report":
    st.title("Research Report")
    results = st.session_state.results
    if not results or not results.get("metrics"):
        st.warning("Run a benchmark first to generate a report.")
        st.stop()

    md = generate_markdown_report(
        results,
        dataset_name=st.session_state.dataset_name,
        original_df=st.session_state.dataset,
    )
    st.markdown(md)

    st.download_button(
        "Download report (Markdown)",
        md,
        file_name="hallubench_report.md",
        mime="text/markdown",
    )
    # Also offer plain text
    st.download_button(
        "Download report (plain text)",
        md,
        file_name="hallubench_report.txt",
        mime="text/plain",
    )
