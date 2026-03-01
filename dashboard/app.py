"""
app.py — Churn Intelligence Platform
-------------------------------------
Streamlit multi-page app entrypoint.
Loads model artifacts once (cached) and exposes them via st.session_state
so all pages can access them without reloading.
"""

import sys
from pathlib import Path

import streamlit as st

# Make src/ importable from the dashboard/ subdirectory
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model         import load_artifacts
from src.shap_analysis import load_shap_artifacts

# ---------------------------------------------------------------------------
# Page configuration (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Churn Intelligence Platform",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Load resources (cached for the entire server session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model artifacts ...")
def load_model_resources():
    model, preprocessor, metadata = load_artifacts(
        model_path=ROOT / "models/xgb_churn_model.joblib",
        preprocessor_path=ROOT / "models/preprocessor.joblib",
        metadata_path=ROOT / "models/model_metadata.json",
    )
    try:
        explainer, shap_values = load_shap_artifacts(
            load_path=ROOT / "reports/shap_values.joblib"
        )
    except FileNotFoundError:
        explainer, shap_values = None, None
    return model, preprocessor, metadata, explainer, shap_values


try:
    model, preprocessor, metadata, explainer, shap_values = load_model_resources()
    st.session_state["model"]        = model
    st.session_state["preprocessor"] = preprocessor
    st.session_state["metadata"]     = metadata
    st.session_state["explainer"]    = explainer
    st.session_state["shap_values"]  = shap_values
    st.session_state["feature_names"] = metadata.get("feature_names", [])
    model_loaded = True
except FileNotFoundError as e:
    model_loaded = False
    load_error   = str(e)

# ---------------------------------------------------------------------------
# Sidebar — shared across all pages
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 📉 Churn Intelligence")
    st.markdown("---")

    if model_loaded:
        m = metadata["metrics"]
        st.metric("AUC-ROC",  f"{m['auc_roc']:.4f}")
        st.metric("F1-Score", f"{m['f1_score']:.4f}")
        st.metric("Recall",   f"{m['recall']:.4f}")
        st.markdown("---")
        st.caption(f"Model v{metadata.get('model_version','1.0.0')}")
        trained_at = metadata.get("trained_at", "")
        if trained_at:
            st.caption(f"Trained: {trained_at[:10]}")
        st.caption(f"XGBoost {metadata.get('xgboost_version','')}")
    else:
        st.error("Model not trained yet.")
        st.code("python train.py", language="bash")

    st.markdown("---")
    st.markdown(
        "Built with XGBoost + SHAP + Streamlit\n\n"
        "[View on GitHub](https://github.com)"
    )

# ---------------------------------------------------------------------------
# Home page content
# ---------------------------------------------------------------------------

st.title("📉 Churn Intelligence Platform")
st.markdown(
    "**Predict. Explain. Retain.** — An ML-powered platform to identify "
    "customers at risk of churning and generate actionable business insights."
)

if not model_loaded:
    st.error(f"Model artifacts not found.\n\n```\n{load_error}\n```")
    st.info("Run `python train.py` from the project root, then refresh this page.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)

m = metadata["metrics"]
col1.metric("AUC-ROC",    f"{m['auc_roc']:.3f}",   help="Area under ROC curve")
col2.metric("F1-Score",   f"{m['f1_score']:.3f}",  help="Harmonic mean of precision & recall")
col3.metric("Precision",  f"{m['precision']:.3f}", help="Of predicted churners, how many were correct")
col4.metric("Recall",     f"{m['recall']:.3f}",    help="Of actual churners, how many did we catch")

st.markdown("---")

st.markdown("""
### Navigation

Use the sidebar pages to explore the platform:

| Page | Description |
|------|-------------|
| **1 Executive Overview** | Model KPIs, confusion matrix, ROC curve, business impact |
| **2 Customer Risk Analyzer** | Score a single customer and explain the prediction with SHAP |
| **3 Feature Importance** | Global SHAP analysis — what drives churn most? |
| **4 Business Calculator** | ROI calculator for your retention program |
""")

st.markdown("---")
st.info(
    "**Dataset:** IBM Telco Customer Churn — 7,043 real telecom customers, 21 features.  "
    "**Algorithm:** XGBoost with SHAP TreeExplainer for exact Shapley values."
)
