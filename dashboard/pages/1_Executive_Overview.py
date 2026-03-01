"""
Page 1 — Executive Overview
Model KPIs, confusion matrix, ROC curve, and business impact summary.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Executive Overview", page_icon="📊", layout="wide")

# ---------------------------------------------------------------------------
# Guard: model must be loaded from app.py session_state
# ---------------------------------------------------------------------------

if "metadata" not in st.session_state:
    st.error("Please open the app from the main page (dashboard/app.py).")
    st.stop()

metadata = st.session_state["metadata"]
m        = metadata["metrics"]

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------

st.title("📊 Executive Overview")
st.markdown("Model performance summary and business impact analysis.")

# ---------------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)
col1.metric("AUC-ROC",   f"{m['auc_roc']:.4f}",   delta="↑ vs. random (0.50)")
col2.metric("F1-Score",  f"{m['f1_score']:.4f}")
col3.metric("Precision", f"{m['precision']:.4f}",  help="How accurate are our churn alerts?")
col4.metric("Recall",    f"{m['recall']:.4f}",     help="What % of churners do we catch?")

st.markdown("---")

# ---------------------------------------------------------------------------
# Confusion matrix + ROC curve side by side
# ---------------------------------------------------------------------------

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Confusion Matrix")
    cm_path = ROOT / "reports/figures/confusion_matrix.png"
    if cm_path.exists():
        st.image(str(cm_path), use_container_width=True)
    else:
        # Re-draw from metrics
        tn = m["tn"]; fp = m["fp"]; fn = m["fn"]; tp = m["tp"]
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            np.array([[tn, fp], [fn, tp]]),
            annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"],
            ax=ax,
        )
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

with col_right:
    st.subheader("ROC Curve")
    roc_path = ROOT / "reports/figures/roc_curve.png"
    if roc_path.exists():
        st.image(str(roc_path), use_container_width=True)
    else:
        st.info("ROC curve image not found. Re-run train.py.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Business Impact Calculator (sidebar sliders)
# ---------------------------------------------------------------------------

st.subheader("💰 Business Impact")

with st.sidebar:
    st.markdown("### Business Parameters")
    avg_monthly_revenue    = st.number_input("Avg monthly revenue per customer ($)", value=65.0, step=1.0)
    retention_cost         = st.number_input("Retention cost per customer ($)",       value=15.0, step=1.0)
    retention_success_rate = st.slider("Retention success rate (%)", 10, 80, 40) / 100

tn = m["tn"]; fp = m["fp"]; fn = m["fn"]; tp = m["tp"]
total_test     = tn + fp + fn + tp
flagged        = tp + fp
revenue_risk   = (tp + fn) * avg_monthly_revenue
revenue_saved  = tp * retention_success_rate * avg_monthly_revenue
program_cost   = flagged * retention_cost
net_benefit    = revenue_saved - program_cost
roi_pct        = (net_benefit / max(program_cost, 1)) * 100

bi1, bi2, bi3, bi4 = st.columns(4)
bi1.metric("Customers in Test Set",      f"{total_test:,}")
bi2.metric("Revenue at Risk (monthly)",  f"${revenue_risk:,.0f}", help="Actual churners × avg revenue")
bi3.metric("Revenue Saved (monthly)",    f"${revenue_saved:,.0f}")
bi4.metric("Program ROI",                f"{roi_pct:.0f}%",
           delta=f"Net ${net_benefit:,.0f}" if net_benefit >= 0 else f"-${abs(net_benefit):,.0f}",
           delta_color="normal")

st.markdown("---")

# ---------------------------------------------------------------------------
# Full classification report
# ---------------------------------------------------------------------------

with st.expander("📋 Full Classification Report"):
    st.code(m.get("classification_report", "Not available"), language="text")

# ---------------------------------------------------------------------------
# Model info
# ---------------------------------------------------------------------------

with st.expander("⚙️ Model Technical Details"):
    info = {
        "Model Version":     metadata.get("model_version"),
        "Trained At":        metadata.get("trained_at", "")[:19],
        "XGBoost Version":   metadata.get("xgboost_version"),
        "scikit-learn":      metadata.get("sklearn_version"),
        "Best Iteration":    m.get("best_iteration"),
        "Threshold":         m.get("threshold"),
        "Features":          len(metadata.get("feature_names", [])),
    }
    st.table(pd.DataFrame.from_dict(info, orient="index", columns=["Value"]))
