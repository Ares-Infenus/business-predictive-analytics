"""
Page 2 — Customer Risk Analyzer
Input a customer's details and get a real-time churn prediction + SHAP waterfall.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.model         import predict_single_customer
from src.shap_analysis import plot_shap_waterfall_single
from src.feature_engineering import clean_raw_data

st.set_page_config(page_title="Customer Risk Analyzer", page_icon="🔍", layout="wide")

# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------

if "model" not in st.session_state:
    st.error("Please open the app via dashboard/app.py first.")
    st.stop()

model        = st.session_state["model"]
preprocessor = st.session_state["preprocessor"]
explainer    = st.session_state["explainer"]
feature_names = st.session_state.get("feature_names", [])

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------

st.title("🔍 Customer Risk Analyzer")
st.markdown(
    "Enter a customer's profile below to get their **churn probability** "
    "and a SHAP explanation of what drives the prediction."
)

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------

with st.form("customer_form"):
    st.subheader("Customer Profile")

    r1c1, r1c2, r1c3 = st.columns(3)
    gender        = r1c1.selectbox("Gender",      ["Male", "Female"])
    senior_raw    = r1c2.radio("Senior Citizen",  ["No (0)", "Yes (1)"], horizontal=True)
    partner       = r1c3.selectbox("Partner",     ["Yes", "No"])

    r2c1, r2c2, r2c3 = st.columns(3)
    dependents    = r2c1.selectbox("Dependents",  ["Yes", "No"])
    tenure        = r2c2.slider("Tenure (months)", 0, 72, 12)
    phone_service = r2c3.selectbox("Phone Service", ["Yes", "No"])

    r3c1, r3c2, r3c3 = st.columns(3)
    multiple_lines   = r3c1.selectbox("Multiple Lines",    ["No", "Yes", "No phone service"])
    internet_service = r3c2.selectbox("Internet Service",  ["DSL", "Fiber optic", "No"])
    contract         = r3c3.selectbox("Contract",          ["Month-to-month", "One year", "Two year"])

    r4c1, r4c2, r4c3 = st.columns(3)
    monthly_charges = r4c1.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0,
                                        value=65.0, step=0.5)
    total_charges   = r4c2.number_input("Total Charges ($)",   min_value=0.0,  max_value=9000.0,
                                        value=float(tenure * 65), step=10.0)
    payment_method  = r4c3.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

    st.markdown("**Internet Add-ons**")
    r5c1, r5c2, r5c3 = st.columns(3)
    no_internet = internet_service == "No"
    addon_opts  = ["No internet service"] if no_internet else ["Yes", "No"]

    online_security  = r5c1.selectbox("Online Security",   addon_opts)
    online_backup    = r5c2.selectbox("Online Backup",     addon_opts)
    device_prot      = r5c3.selectbox("Device Protection", addon_opts)

    r6c1, r6c2, r6c3 = st.columns(3)
    tech_support     = r6c1.selectbox("Tech Support",      addon_opts)
    streaming_tv     = r6c2.selectbox("Streaming TV",      addon_opts)
    streaming_movies = r6c3.selectbox("Streaming Movies",  addon_opts)

    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

    submitted = st.form_submit_button("🔮 Analyze Churn Risk", use_container_width=True)

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

if submitted:
    senior_int = 1 if "Yes" in senior_raw else 0

    customer_dict = {
        "gender":          gender,
        "SeniorCitizen":   senior_int,
        "Partner":         partner,
        "Dependents":      dependents,
        "tenure":          tenure,
        "PhoneService":    phone_service,
        "MultipleLines":   multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity":  online_security,
        "OnlineBackup":    online_backup,
        "DeviceProtection":device_prot,
        "TechSupport":     tech_support,
        "StreamingTV":     streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract":        contract,
        "PaperlessBilling":paperless,
        "PaymentMethod":   payment_method,
        "MonthlyCharges":  monthly_charges,
        "TotalCharges":    str(total_charges),
    }

    try:
        # Clean and preprocess for prediction
        df_single = pd.DataFrame([customer_dict])
        # Clean TotalCharges
        df_single["TotalCharges"] = pd.to_numeric(
            df_single["TotalCharges"], errors="coerce"
        ).fillna(tenure * monthly_charges)
        df_single["SeniorCitizen"] = df_single["SeniorCitizen"].astype(int)

        X_proc = preprocessor.transform(df_single)
        proba  = float(model.predict_proba(X_proc)[0, 1])

        # Result display
        st.markdown("---")
        res1, res2, res3 = st.columns(3)

        res1.metric("Churn Probability", f"{proba:.1%}")

        if proba < 0.30:
            risk_tier = "LOW"
            res2.success(f"Risk Tier: {risk_tier}")
            res3.success("Recommended Action: Standard retention program is sufficient.")
        elif proba < 0.60:
            risk_tier = "MEDIUM"
            res2.warning(f"Risk Tier: {risk_tier}")
            res3.warning("Recommended Action: Schedule a check-in call within 2 weeks and send a satisfaction survey.")
        else:
            risk_tier = "HIGH"
            res2.error(f"Risk Tier: {risk_tier}")
            res3.error("Recommended Action: Immediate outreach — offer a loyalty discount or contract upgrade.")

        # SHAP waterfall
        if explainer is not None:
            st.markdown("---")
            st.subheader("🔬 SHAP Explanation")
            st.markdown(
                "The chart below shows which features **pushed this customer's churn risk UP (red)** "
                "or **DOWN (blue)** relative to the average customer."
            )
            try:
                fig = plot_shap_waterfall_single(
                    explainer, X_proc, feature_names
                )
                st.pyplot(fig)
            except Exception as shap_err:
                st.warning(f"Could not render SHAP waterfall: {shap_err}")
        else:
            st.info("SHAP explainer not available. Re-run train.py without --no-shap.")

    except Exception as err:
        st.error(f"Prediction error: {err}")
        st.exception(err)

else:
    st.info("Fill in the customer profile above and click **Analyze Churn Risk**.")
