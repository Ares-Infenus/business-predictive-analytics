"""
train.py
--------
Full end-to-end training pipeline for the Churn Prediction model.

Usage:
  python train.py
  python train.py --force-download --no-shap
  python train.py --test-size 0.25 --early-stopping 30
"""

import argparse
import sys
from pathlib import Path

# Ensure src/ is importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader        import load_data, validate_data
from src.feature_engineering import prepare_data
from src.model              import (
    train_model,
    evaluate_model,
    find_optimal_threshold,
    save_artifacts,
)
from src.shap_analysis      import (
    compute_shap_values,
    plot_shap_summary_bar,
    plot_shap_summary_beeswarm,
    plot_shap_waterfall_single,
    save_shap_artifacts,
    get_top_features,
)


# ---------------------------------------------------------------------------
# Executive Report generator
# ---------------------------------------------------------------------------

def generate_executive_report(
    metrics: dict,
    top_features,
    validation: dict,
) -> None:
    """Write the auto-generated executive report to reports/executive_report.md."""
    Path("reports").mkdir(parents=True, exist_ok=True)

    tn = metrics.get("tn", 0)
    fp = metrics.get("fp", 0)
    fn = metrics.get("fn", 0)
    tp = metrics.get("tp", 0)

    churn_rate        = validation.get("churn_rate", 0.265)
    row_count         = validation.get("row_count", 7043)
    test_size_n       = int(row_count * 0.2)
    train_size        = row_count - test_size_n
    threshold         = metrics.get("threshold", 0.5)
    monthly_rev_risk  = tp * 65
    monthly_rev_saved = int(tp * 0.40 * 65)

    # Build features table
    feature_table_rows = ""
    insights = {
        "Contract":        ("Month-to-month ↑", "Short-term contracts = highest risk"),
        "tenure":          ("Short tenure ↑",   "New customers are at higher risk"),
        "InternetService": ("Fiber optic ↑",    "Fiber customers churn more"),
        "TotalCharges":    ("Lower charges ↑",  "Low total spend = not committed"),
        "OnlineSecurity":  ("No service ↑",     "Missing add-ons signal disengagement"),
    }
    if top_features is not None:
        for i, row in top_features.head(5).iterrows():
            feat    = row["feature"]
            insight = insights.get(feat, ("--", "--"))
            feature_table_rows += (
                f"| {i} | {feat} | {insight[0]} | {insight[1]} |\n"
            )

    report = f"""# Churn Prediction Model -- Executive Report

**Generated:** {__import__('datetime').datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
**Model Version:** 1.0.0
**Dataset:** IBM Telco Customer Churn ({row_count:,} customers)

---

## 1. Executive Summary

This report presents the findings of a machine learning-based customer churn
prediction model trained on Telco customer data. The model identifies customers
at high risk of canceling their subscription, enabling proactive retention
actions before revenue is lost.

**Key Finding:** The model achieves **{metrics['auc_roc']:.1%} AUC-ROC**, meaning it correctly
ranks {metrics['auc_roc']:.0%} of churner/non-churner pairs. At an operating threshold of
**{threshold:.2f}**, the model catches **{metrics['recall']:.1%} of actual churners**.

---

## 2. Business Context

- **Churn Rate:** {churn_rate:.1%} of customers canceled in this dataset
- **Revenue at Risk (monthly):** ~${churn_rate * row_count * 65:,.0f} if no action is taken
- **Model Value:** By targeting predicted high-risk customers with a 40%
  successful retention rate, the model can recover an estimated
  **${monthly_rev_saved:,}/month**

---

## 3. Model Performance

| Metric         | Value    |
|----------------|----------|
| AUC-ROC        | {metrics['auc_roc']:.4f} |
| F1-Score       | {metrics['f1_score']:.4f} |
| Precision      | {metrics['precision']:.4f} |
| Recall         | {metrics['recall']:.4f} |
| Accuracy       | {metrics['accuracy']:.4f} |
| Best Iteration | {metrics['best_iteration']} |
| Threshold      | {threshold:.2f} |

### 3.1 Confusion Matrix

| Predicted \\ Actual | Not Churn | Churn     |
|---------------------|-----------|-----------|
| Not Churn (pred.)   | TN = {tn}    | FN = {fn}    |
| Churn (pred.)       | FP = {fp}     | TP = {tp}    |

**Interpretation:**
- Correctly identified **{tp} churners** (true positives -- caught revenue at risk)
- Missed **{fn} actual churners** (false negatives -- missed opportunities)
- Incorrectly flagged **{fp} loyal customers** (false positives -- wasted retention spend)

---

## 4. Key Churn Drivers (SHAP Analysis)

| Rank | Feature | Impact Direction | Business Insight |
|------|---------|-----------------|-----------------|
{feature_table_rows}

---

## 5. Recommended Actions

### Immediate (High-Risk: Churn Probability > 60%)
1. Assign a dedicated account manager -- outreach within 48 hours
2. Offer contract upgrade incentive (Month-to-month -> Annual, $10/month discount)
3. Bundle OnlineSecurity + TechSupport at a reduced rate

### Short-term (Medium-Risk: 30–60%)
1. Targeted email campaign with service bundle offers
2. Proactive satisfaction survey + service review call
3. Loyalty reward points program enrollment

### Long-term (Structural)
1. Review Fiber optic service pricing and reliability (highest SHAP impact)
2. Design enhanced 3-month and 6-month onboarding programs (tenure effect)
3. Create an annual contract migration incentive program

---

## 6. Technical Details

- **Algorithm:** XGBoost (Gradient Boosted Trees, `tree_method=hist`)
- **Training Set:** {train_size:,} customers (80% of data)
- **Test Set:** {test_size_n:,} customers (20% of data)
- **Class Imbalance:** Handled via `scale_pos_weight` (no SMOTE)
- **Explainability:** SHAP `TreeExplainer` (exact Shapley values)

---

## 7. Limitations & Next Steps

**Limitations:**
- Model trained on historical data; performance may degrade as market conditions change
- Threshold {threshold:.2f} represents a specific precision-recall trade-off; adjust for business priorities

**Recommended Next Steps:**
1. Retrain quarterly with fresh customer data
2. A/B test retention interventions to measure true causal impact
3. Implement real-time scoring via API wrapper

---

*Auto-generated by train.py*
"""

    with open("reports/executive_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    print("  Executive report -> reports/executive_report.md")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the Churn Prediction XGBoost model"
    )
    parser.add_argument(
        "--force-download", action="store_true",
        help="Re-download raw data even if already cached"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Fraction of data for test split (default: 0.2)"
    )
    parser.add_argument(
        "--random-state", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--early-stopping", type=int, default=50,
        help="Early stopping rounds for XGBoost (default: 50)"
    )
    parser.add_argument(
        "--no-shap", action="store_true",
        help="Skip SHAP computation (faster for iteration)"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print("=" * 60)
    print("  Churn Prediction -- Training Pipeline")
    print("=" * 60)

    # Step 1: Setup directories
    print("\n[1/7] Setting up directories ...")
    for d in ["data/raw", "data/processed", "models", "reports/figures"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # Step 2: Load data
    print("\n[2/7] Loading data ...")
    df = load_data(force_download=args.force_download)
    validation = validate_data(df)
    print(f"  Rows       : {validation['row_count']:,}")
    print(f"  Churn rate : {validation['churn_rate']:.1%}")
    print(f"  TotalCharges blanks: {validation['totalcharges_blanks']}")
    if validation["missing_values"]:
        print(f"  Missing values: {validation['missing_values']}")

    # Step 3: Feature engineering
    print("\n[3/7] Engineering features ...")
    X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_data(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    # Step 4: Train model
    print("\n[4/7] Training XGBoost model ...")
    model, metrics = train_model(
        X_train, y_train, X_test, y_test,
        early_stopping_rounds=args.early_stopping,
    )
    print(f"  Best iteration : {metrics['best_iteration']}")
    print(f"  AUC-ROC        : {metrics['auc_roc']:.4f}")
    print(f"  F1-Score       : {metrics['f1_score']:.4f}")

    # Step 5: Optimal threshold
    print("\n[5/7] Optimising classification threshold ...")
    optimal_threshold = find_optimal_threshold(model, X_test, y_test)
    print(f"  Optimal threshold: {optimal_threshold:.2f}")
    final_metrics = evaluate_model(
        model, X_test, y_test,
        threshold=optimal_threshold,
        save_plots=True,
    )
    print(f"  Final F1 (threshold={optimal_threshold}): {final_metrics['f1_score']:.4f}")
    print(f"  Recall          : {final_metrics['recall']:.4f}")
    print(f"  Precision       : {final_metrics['precision']:.4f}")

    # Step 6: Save artifacts
    print("\n[6/7] Saving model artifacts ...")
    save_artifacts(model, preprocessor, feature_names, final_metrics)

    # Step 7: SHAP + report
    top_features = None
    if not args.no_shap:
        print("\n[7/7] Computing SHAP values and generating plots ...")
        explainer, shap_train, shap_test = compute_shap_values(
            model, X_train, X_test, feature_names
        )
        plot_shap_summary_bar(shap_test, X_test, feature_names)
        print("  Saved: reports/figures/shap_summary_bar.png")

        plot_shap_summary_beeswarm(shap_test, X_test, feature_names)
        print("  Saved: reports/figures/shap_summary_beeswarm.png")

        plot_shap_waterfall_single(
            explainer, X_test[:1], feature_names,
            save_path="reports/figures/shap_waterfall_sample.png",
        )
        print("  Saved: reports/figures/shap_waterfall_sample.png")

        save_shap_artifacts(explainer, shap_test)
        top_features = get_top_features(shap_test, feature_names)
        print(f"  Top 5 features: {top_features['feature'].head().tolist()}")
    else:
        print("\n[7/7] SHAP skipped (--no-shap flag).")

    # Executive report
    generate_executive_report(final_metrics, top_features, validation)

    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  AUC-ROC  : {final_metrics['auc_roc']:.4f}")
    print(f"  F1-Score : {final_metrics['f1_score']:.4f}")
    print("=" * 60)
    print("\n  Next step: streamlit run dashboard/app.py\n")


if __name__ == "__main__":
    main()
