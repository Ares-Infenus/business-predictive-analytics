"""
Page 3 — Feature Importance
Global SHAP analysis: which features drive churn most?
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.shap_analysis import (
    plot_shap_summary_bar,
    plot_shap_summary_beeswarm,
    plot_shap_dependence,
    get_top_features,
)

st.set_page_config(page_title="Feature Importance", page_icon="📈", layout="wide")

# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------

if "metadata" not in st.session_state:
    st.error("Please open the app via dashboard/app.py first.")
    st.stop()

explainer    = st.session_state.get("explainer")
shap_values  = st.session_state.get("shap_values")
feature_names = st.session_state.get("feature_names", [])

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------

st.title("📈 Feature Importance (SHAP)")
st.markdown(
    "This page uses **SHAP (SHapley Additive exPlanations)** to quantify "
    "the contribution of each feature to churn predictions across all customers."
)

if shap_values is None or explainer is None:
    st.warning("SHAP artifacts not found. Re-run `python train.py` (without `--no-shap`).")
    st.stop()

# Load X_test to display dependence plots (saved as part of shap artifacts)
shap_data_path = ROOT / "reports/shap_values.joblib"
try:
    raw = joblib.load(shap_data_path)
    # X_test_proc was not saved separately — we'll use shap_values shape to indicate it's available
except Exception:
    pass

# ---------------------------------------------------------------------------
# Top N slider
# ---------------------------------------------------------------------------

top_n = st.sidebar.slider("Features to display", min_value=5, max_value=30, value=20)

# ---------------------------------------------------------------------------
# Summary bar chart
# ---------------------------------------------------------------------------

st.subheader("Global Feature Importance (Mean |SHAP|)")

bar_path = ROOT / "reports/figures/shap_summary_bar.png"
if bar_path.exists():
    st.image(str(bar_path), use_container_width=True)
else:
    with st.spinner("Generating SHAP summary bar chart ..."):
        # We need X_proc — try to reconstruct from data
        st.info("Image not cached. Re-run train.py to pre-generate plots.")

# ---------------------------------------------------------------------------
# Beeswarm plot
# ---------------------------------------------------------------------------

st.subheader("Feature Impact Distribution (Beeswarm)")
st.markdown(
    "Each dot represents one customer. "
    "**Color** = feature value (blue = low, red = high). "
    "**Position** = SHAP value (impact on churn probability)."
)

beeswarm_path = ROOT / "reports/figures/shap_summary_beeswarm.png"
if beeswarm_path.exists():
    st.image(str(beeswarm_path), use_container_width=True)
else:
    st.info("Beeswarm plot not cached. Re-run train.py.")

# ---------------------------------------------------------------------------
# Top features table
# ---------------------------------------------------------------------------

st.subheader("Top Feature Rankings")

top_df = get_top_features(shap_values, feature_names, top_n=top_n)
top_df_display = top_df.copy().reset_index()
top_df_display.columns = ["Rank", "Feature", "Mean |SHAP|", "% of Total Impact"]
top_df_display["Mean |SHAP|"]        = top_df_display["Mean |SHAP|"].round(4)
top_df_display["% of Total Impact"]  = top_df_display["% of Total Impact"].round(1)

st.dataframe(
    top_df_display.style.background_gradient(subset=["Mean |SHAP|"], cmap="YlOrRd"),
    use_container_width=True,
    hide_index=True,
)

# Download button
csv_data = top_df_display.to_csv(index=False).encode("utf-8")
st.download_button(
    "📥 Download Feature Importance CSV",
    data=csv_data,
    file_name="feature_importance_shap.csv",
    mime="text/csv",
)

# ---------------------------------------------------------------------------
# Business interpretation callouts
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("💡 Business Insights")

insights = {
    "Contract":         "Month-to-month contracts have the highest churn risk. "
                        "Incentivise customers to switch to annual contracts.",
    "tenure":           "New customers (low tenure) are disproportionately likely to churn. "
                        "Invest in strong onboarding programs for the first 6 months.",
    "InternetService":  "Fiber optic customers churn more — possibly due to price or service quality. "
                        "Review Fiber pricing and support SLAs.",
    "TotalCharges":     "Lower total charges (newer or lower-spend customers) signal higher churn risk. "
                        "Target low-spend segments with value-add offers.",
    "OnlineSecurity":   "Customers without Online Security or Tech Support are more likely to churn. "
                        "Bundle these add-ons as free trial offers.",
}

top_feature_names = top_df["feature"].head(5).tolist()
for feat in top_feature_names:
    for key, insight in insights.items():
        if key.lower() in feat.lower():
            st.info(f"**{feat}:** {insight}")
            break
