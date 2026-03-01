"""
shap_analysis.py
----------------
Compute SHAP values and generate interpretability plots for the
XGBoost churn model. All plotting functions return matplotlib Figure
objects suitable for direct use in Streamlit.
"""

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

FIGURES_PATH   = Path("reports/figures")
SHAP_SAVE_PATH = Path("reports/shap_values.joblib")


# ---------------------------------------------------------------------------
# Compute SHAP values
# ---------------------------------------------------------------------------

def compute_shap_values(
    model,
    X_train_proc: np.ndarray,
    X_test_proc: np.ndarray,
    feature_names: list,
) -> tuple:
    """
    Compute exact SHAP values using TreeExplainer.

    Returns:
      (explainer, shap_values_train, shap_values_test)
      shap_values arrays have shape (n_samples, n_features).
    """
    print("  Initialising TreeExplainer ...")
    explainer = shap.TreeExplainer(model)

    print("  Computing SHAP values for training set ...")
    shap_values_train = explainer.shap_values(X_train_proc)

    print("  Computing SHAP values for test set ...")
    shap_values_test  = explainer.shap_values(X_test_proc)

    return explainer, shap_values_train, shap_values_test


# ---------------------------------------------------------------------------
# Summary — bar chart
# ---------------------------------------------------------------------------

def plot_shap_summary_bar(
    shap_values: np.ndarray,
    X_proc: np.ndarray,
    feature_names: list,
    save_path: str = None,
    top_n: int = 20,
) -> plt.Figure:
    """
    Bar chart of mean |SHAP| per feature (global importance).
    Saves PNG if save_path is provided.
    Returns matplotlib Figure.
    """
    if save_path is None:
        save_path = str(FIGURES_PATH / "shap_summary_bar.png")
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_proc,
        feature_names=feature_names,
        plot_type="bar",
        max_display=top_n,
        show=False,
    )
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Summary — beeswarm
# ---------------------------------------------------------------------------

def plot_shap_summary_beeswarm(
    shap_values: np.ndarray,
    X_proc: np.ndarray,
    feature_names: list,
    save_path: str = None,
    top_n: int = 20,
) -> plt.Figure:
    """
    Beeswarm plot showing direction + magnitude of feature impact.
    Saves PNG if save_path is provided.
    Returns matplotlib Figure.
    """
    if save_path is None:
        save_path = str(FIGURES_PATH / "shap_summary_beeswarm.png")
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_proc,
        feature_names=feature_names,
        max_display=top_n,
        show=False,
    )
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Waterfall — single customer
# ---------------------------------------------------------------------------

def plot_shap_waterfall_single(
    explainer,
    X_single: np.ndarray,
    feature_names: list,
    customer_id: str = "sample",
    save_path: str = None,
) -> plt.Figure:
    """
    Waterfall plot explaining a single customer's churn prediction.
    Shows which features pushed the score up or down from the base value.

    Returns matplotlib Figure (does NOT call plt.show()).
    """
    # Build Explanation object
    sv = explainer(X_single)

    # Handle both 2-D shap_values output (binary classification) and Explanation
    if hasattr(sv, "values") and sv.values.ndim == 2 and sv.values.shape[0] == 1:
        # Single row Explanation
        explanation = shap.Explanation(
            values=sv.values[0],
            base_values=sv.base_values[0] if sv.base_values.ndim > 0 else sv.base_values,
            data=sv.data[0] if sv.data is not None else None,
            feature_names=feature_names,
        )
    else:
        explanation = sv[0]
        explanation.feature_names = feature_names

    plt.figure(figsize=(10, 7))
    shap.plots.waterfall(explanation, show=False)
    fig = plt.gcf()
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Dependence plot
# ---------------------------------------------------------------------------

def plot_shap_dependence(
    shap_values: np.ndarray,
    X_proc: np.ndarray,
    feature_names: list,
    feature: str,
    interaction_feature: str = "auto",
    save_path: str = None,
) -> plt.Figure:
    """
    Dependence plot for a single feature showing its SHAP value vs. raw value.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.dependence_plot(
        feature, shap_values, X_proc,
        feature_names=feature_names,
        interaction_index=interaction_feature,
        ax=ax,
        show=False,
    )
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Feature importance table
# ---------------------------------------------------------------------------

def get_top_features(
    shap_values: np.ndarray,
    feature_names: list,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Compute mean(|SHAP|) per feature.
    Returns a DataFrame with columns [feature, mean_abs_shap]
    sorted descending, limited to top_n rows.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    total    = mean_abs.sum()

    df = pd.DataFrame({
        "feature":       feature_names,
        "mean_abs_shap": mean_abs,
    })
    df["pct_impact"] = df["mean_abs_shap"] / total * 100
    df = df.sort_values("mean_abs_shap", ascending=False).head(top_n).reset_index(drop=True)
    df.index += 1  # 1-based rank
    df.index.name = "rank"

    return df


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def save_shap_artifacts(
    explainer,
    shap_values_test: np.ndarray,
    save_path: Path = SHAP_SAVE_PATH,
) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"explainer": explainer, "shap_values": shap_values_test}, save_path)
    print(f"  SHAP artifacts -> {save_path}")


def load_shap_artifacts(load_path: Path = SHAP_SAVE_PATH) -> tuple:
    """Load and return (explainer, shap_values_test)."""
    if not Path(load_path).exists():
        raise FileNotFoundError(
            f"SHAP artifacts not found at {load_path}.\n"
            "Run `python train.py` first."
        )
    data = joblib.load(load_path)
    return data["explainer"], data["shap_values"]
