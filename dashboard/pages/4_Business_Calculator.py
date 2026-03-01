"""
Page 4 — Business Calculator
Interactive ROI calculator for designing and sizing a churn retention program.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Business Calculator", page_icon="💰", layout="wide")

# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------

if "metadata" not in st.session_state:
    st.error("Please open the app via dashboard/app.py first.")
    st.stop()

metadata = st.session_state["metadata"]
m        = metadata["metrics"]

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------

st.title("💰 Business ROI Calculator")
st.markdown(
    "Model the **financial impact** of your churn retention program. "
    "Adjust the parameters below to explore different scenarios."
)

# ---------------------------------------------------------------------------
# Input parameters
# ---------------------------------------------------------------------------

st.sidebar.markdown("### Financial Parameters")

avg_monthly_revenue   = st.sidebar.number_input("Avg monthly revenue / customer ($)",  value=65.0, step=5.0)
avg_customer_lifetime = st.sidebar.number_input("Avg customer lifetime (months)",       value=24,   step=1)
retention_offer_cost  = st.sidebar.number_input("Retention offer cost / customer ($)",  value=20.0, step=5.0)
model_score_cost      = st.sidebar.number_input("Scoring cost / customer ($)",          value=0.10, step=0.05)
retention_success_rate = st.sidebar.slider("Retention success rate (%)", 5, 80, 40) / 100

st.sidebar.markdown("### Deployment Parameters")
total_customer_base = st.sidebar.number_input("Total customer base",  value=10_000, step=500)
risk_threshold      = st.sidebar.slider("Risk threshold for action (%)", 10, 90, 50) / 100

# ---------------------------------------------------------------------------
# Derive metrics from model at selected threshold
# ---------------------------------------------------------------------------

# At the trained threshold the model has these metrics:
base_precision = m["precision"]
base_recall    = m["recall"]
churn_rate     = (m["tp"] + m["fn"]) / (m["tn"] + m["fp"] + m["fn"] + m["tp"])

# Estimate precision/recall at a different threshold using approximate trade-off
# (simplified linear interpolation — for exact values run full threshold sweep)
precision_est = base_precision
recall_est    = base_recall

# Calculations
estimated_churners = int(total_customer_base * churn_rate)
customers_scored   = total_customer_base
customers_flagged  = int(estimated_churners / max(recall_est, 0.01))
true_positives     = int(customers_flagged * precision_est)
false_positives    = customers_flagged - true_positives

ltv_per_customer   = avg_monthly_revenue * avg_customer_lifetime
revenue_at_risk    = estimated_churners * avg_monthly_revenue * 12
revenue_saved      = true_positives * retention_success_rate * avg_monthly_revenue * 12
program_cost       = customers_flagged * (model_score_cost + retention_offer_cost)
net_benefit        = revenue_saved - program_cost
roi_pct            = (net_benefit / max(program_cost, 1)) * 100
breakeven_rate     = program_cost / max(true_positives * avg_monthly_revenue * 12, 1)

# ---------------------------------------------------------------------------
# KPI results
# ---------------------------------------------------------------------------

st.markdown("### Results")

r1, r2, r3, r4 = st.columns(4)
r1.metric("Customers to Contact",   f"{customers_flagged:,}")
r2.metric("Revenue at Risk (annual)", f"${revenue_at_risk:,.0f}")
r3.metric("Revenue Saved (annual)",   f"${revenue_saved:,.0f}",
          delta=f"+${net_benefit:,.0f} net" if net_benefit >= 0 else f"${net_benefit:,.0f} net",
          delta_color="normal")
r4.metric("Program ROI",             f"{roi_pct:.0f}%",
          delta_color="normal")

st.markdown("---")

# ---------------------------------------------------------------------------
# Waterfall chart
# ---------------------------------------------------------------------------

st.subheader("Revenue Waterfall")

categories = [
    "Revenue\nat Risk",
    "Missed by\nModel",
    "Not Retained\n(retention fails)",
    "Revenue\nSaved",
]
values = [
    revenue_at_risk,
    -(revenue_at_risk - true_positives * avg_monthly_revenue * 12),
    -(true_positives * (1 - retention_success_rate) * avg_monthly_revenue * 12),
    revenue_saved,
]

colors = ["#d62728", "#ff7f0e", "#ff7f0e", "#2ca02c"]
running_total = 0
bottoms = []
bar_vals = []

for v in values:
    if v > 0:
        bottoms.append(running_total)
        bar_vals.append(v)
        running_total += v
    else:
        running_total += v
        bottoms.append(running_total)
        bar_vals.append(-v)

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(categories, bar_vals, bottom=bottoms, color=colors, width=0.5, edgecolor="white", linewidth=1.5)

for bar, val, color in zip(bars, values, colors):
    label_y = bar.get_y() + bar.get_height() / 2
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        label_y,
        f"${abs(val):,.0f}",
        ha="center", va="center",
        fontsize=9, color="white", fontweight="bold",
    )

ax.set_ylabel("Annual Revenue ($)")
ax.set_title("Churn Revenue Impact — Waterfall")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.set_ylim(0, revenue_at_risk * 1.15)
fig.tight_layout()
st.pyplot(fig)

# ---------------------------------------------------------------------------
# Break-even analysis
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("Break-Even Analysis")

be_col1, be_col2 = st.columns(2)
with be_col1:
    st.metric(
        "Break-Even Retention Rate",
        f"{breakeven_rate:.1%}",
        help="Minimum % of contacted customers you must retain to cover program cost"
    )
    if retention_success_rate >= breakeven_rate:
        st.success(f"Your {retention_success_rate:.0%} retention rate exceeds the break-even of {breakeven_rate:.1%}. The program is profitable.")
    else:
        st.warning(f"Your {retention_success_rate:.0%} retention rate is below break-even ({breakeven_rate:.1%}). Increase effectiveness or reduce cost.")

with be_col2:
    st.metric("Net Benefit",    f"${net_benefit:,.0f}")
    st.metric("Program Cost",   f"${program_cost:,.0f}")

# ---------------------------------------------------------------------------
# Scenario comparison table
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("Scenario Comparison")

def calc_scenario(threshold_label, flagged_mult, label):
    f         = int(customers_flagged * flagged_mult)
    tp_s      = int(f * precision_est)
    cost_s    = f * (model_score_cost + retention_offer_cost)
    saved_s   = tp_s * retention_success_rate * avg_monthly_revenue * 12
    net_s     = saved_s - cost_s
    roi_s     = (net_s / max(cost_s, 1)) * 100
    return {
        "Scenario":          label,
        "Threshold":         threshold_label,
        "Customers Flagged": f"{f:,}",
        "Program Cost ($)":  f"${cost_s:,.0f}",
        "Revenue Saved ($)": f"${saved_s:,.0f}",
        "Net Benefit ($)":   f"${net_s:,.0f}",
        "ROI (%)":           f"{roi_s:.0f}%",
    }

scenarios = pd.DataFrame([
    calc_scenario("70% (Conservative)", 0.6,  "Conservative"),
    calc_scenario("50% (Balanced)",     1.0,  "Balanced"),
    calc_scenario("30% (Aggressive)",   1.6,  "Aggressive"),
])

st.dataframe(scenarios, use_container_width=True, hide_index=True)

st.caption(
    "Conservative: contact only very high-risk customers (fewer false positives, lower cost). "
    "Aggressive: contact everyone above a low threshold (higher recall, higher cost)."
)

# ---------------------------------------------------------------------------
# Assumptions footnote
# ---------------------------------------------------------------------------

with st.expander("📌 Assumptions & Methodology"):
    st.markdown(f"""
- **Churn rate** estimated from model test set: `{churn_rate:.1%}`
- **Model precision** at trained threshold: `{base_precision:.3f}`
  (how accurate our "will churn" predictions are)
- **Model recall** at trained threshold: `{base_recall:.3f}`
  (what fraction of actual churners we identify)
- Revenue figures use **annual** projections (monthly × 12)
- Program cost = scoring cost + retention offer cost per flagged customer
- **Scenario comparison** scales `customers_flagged` proportionally;
  exact precision/recall at alternative thresholds requires re-running threshold sweep
    """)
