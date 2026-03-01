"""
model.py
--------
Train, evaluate, save and load the XGBoost churn prediction model.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import xgboost as xgb
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------

DEFAULT_PARAMS = {
    "objective":         "binary:logistic",
    "eval_metric":       "auc",
    "n_estimators":      500,
    "learning_rate":     0.05,
    "max_depth":         6,
    "min_child_weight":  1,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "random_state":      42,
    "n_jobs":            -1,
    "tree_method":       "hist",
}

MODEL_PATH        = Path("models/xgb_churn_model.joblib")
PREPROCESSOR_PATH = Path("models/preprocessor.joblib")
METADATA_PATH     = Path("models/model_metadata.json")
FIGURES_PATH      = Path("reports/figures")


# ---------------------------------------------------------------------------
# Class imbalance
# ---------------------------------------------------------------------------

def compute_class_weight(y_train) -> float:
    """Return neg/pos ratio for scale_pos_weight."""
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    return float(n_neg / n_pos)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    X_train: np.ndarray,
    y_train,
    X_test: np.ndarray,
    y_test,
    params: dict = None,
    early_stopping_rounds: int = 50,
) -> tuple:
    """
    Train XGBoost with early stopping on the test AUC.
    Returns (fitted_model, metrics_dict).
    """
    final_params = {**DEFAULT_PARAMS}
    if params:
        final_params.update(params)

    final_params["scale_pos_weight"]    = compute_class_weight(y_train)
    final_params["early_stopping_rounds"] = early_stopping_rounds

    model = XGBClassifier(**final_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    metrics = evaluate_model(model, X_test, y_test)
    return model, metrics


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: XGBClassifier,
    X_test: np.ndarray,
    y_test,
    threshold: float = 0.5,
    save_plots: bool = True,
) -> dict:
    """
    Compute classification metrics and optionally save plots.
    Returns a metrics dict.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    metrics = {
        "auc_roc":               float(roc_auc_score(y_test, y_proba)),
        "f1_score":              float(f1_score(y_test, y_pred)),
        "accuracy":              float(accuracy_score(y_test, y_pred)),
        "precision":             float(precision_score(y_test, y_pred)),
        "recall":                float(recall_score(y_test, y_pred)),
        "confusion_matrix":      [[int(tn), int(fp)], [int(fn), int(tp)]],
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "classification_report": classification_report(y_test, y_pred),
        "best_iteration":        int(getattr(model, "best_iteration", -1)),
        "threshold":             float(threshold),
    }

    if save_plots:
        FIGURES_PATH.mkdir(parents=True, exist_ok=True)
        _save_confusion_matrix(tn, fp, fn, tp)
        _save_roc_curve(y_test, y_proba, metrics["auc_roc"])

    return metrics


def _save_confusion_matrix(tn, fp, fn, tp):
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_array = np.array([[tn, fp], [fn, tp]])
    sns.heatmap(
        cm_array, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(FIGURES_PATH / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def _save_roc_curve(y_test, y_proba, auc_roc: float):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc_roc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Churn Prediction")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_PATH / "roc_curve.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Threshold optimization
# ---------------------------------------------------------------------------

def find_optimal_threshold(
    model: XGBClassifier,
    X_test: np.ndarray,
    y_test,
) -> float:
    """
    Sweep thresholds [0.1, 0.9] and return the one maximising F1.
    """
    y_proba   = model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.10, 0.91, 0.01)
    best_t, best_f1 = 0.5, 0.0

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        score  = f1_score(y_test, y_pred, zero_division=0)
        if score > best_f1:
            best_f1, best_t = score, t

    return round(float(best_t), 2)


# ---------------------------------------------------------------------------
# Artifacts — save / load
# ---------------------------------------------------------------------------

def save_artifacts(
    model: XGBClassifier,
    preprocessor,
    feature_names: list,
    metrics: dict,
    model_path: Path = MODEL_PATH,
    preprocessor_path: Path = PREPROCESSOR_PATH,
    metadata_path: Path = METADATA_PATH,
) -> None:
    """Save model, preprocessor and JSON metadata."""
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)

    serialisable_metrics = {
        k: v for k, v in metrics.items()
        if k != "classification_report"
    }

    metadata = {
        "model_version":    "1.0.0",
        "trained_at":       datetime.now(timezone.utc).isoformat(),
        "feature_names":    feature_names,
        "metrics":          serialisable_metrics,
        "xgboost_version":  xgb.__version__,
        "sklearn_version":  sklearn.__version__,
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Model        -> {model_path}")
    print(f"  Preprocessor -> {preprocessor_path}")
    print(f"  Metadata     -> {metadata_path}")


def load_artifacts(
    model_path: Path = MODEL_PATH,
    preprocessor_path: Path = PREPROCESSOR_PATH,
    metadata_path: Path = METADATA_PATH,
) -> tuple:
    """
    Load and return (model, preprocessor, metadata).
    Raises FileNotFoundError with helpful message if models are missing.
    """
    for p in [model_path, preprocessor_path, metadata_path]:
        if not Path(p).exists():
            raise FileNotFoundError(
                f"Artifact not found: {p}\n"
                "Run `python train.py` first to train the model."
            )

    model        = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    with open(metadata_path) as f:
        metadata = json.load(f)

    return model, preprocessor, metadata


# ---------------------------------------------------------------------------
# Single-customer prediction (for Streamlit)
# ---------------------------------------------------------------------------

def predict_single_customer(
    customer_dict: dict,
    model: XGBClassifier,
    preprocessor,
) -> tuple:
    """
    Predict churn for one customer given raw feature dict.
    Returns (churn_probability: float, churn_label: int).
    """
    df_single = pd.DataFrame([customer_dict])
    X_proc    = preprocessor.transform(df_single)
    proba     = float(model.predict_proba(X_proc)[0, 1])
    label     = int(proba >= 0.5)
    return proba, label


# Lazy import to avoid circular dependency at module level
import pandas as pd
