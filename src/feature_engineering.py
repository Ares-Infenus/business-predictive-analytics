"""
feature_engineering.py
-----------------------
Build and apply the full preprocessing pipeline for the Telco churn dataset.
Returns train/test splits with processed arrays and a fitted ColumnTransformer.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Feature lists
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

# Binary (Yes/No or Male/Female) -- encode with OrdinalEncoder -> 0/1
CATEGORICAL_BINARY = [
    "gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling",
]

# Multi-class categoricals -- encode with OneHotEncoder
CATEGORICAL_MULTI = [
    "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod",
]

# SeniorCitizen is already int (0/1) -- included in numeric pipeline
# customerID is dropped; Churn is the target (separated before pipeline)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-pipeline cleaning:
      - Drop customerID
      - Convert TotalCharges blank strings to NaN, then fill with
        tenure * MonthlyCharges (semantically correct for new customers)
      - SeniorCitizen is already int -- leave as-is
      - Strip leading/trailing whitespace from all object columns
    """
    df = df.copy()

    # Drop identifier
    df = df.drop(columns=["customerID"], errors="ignore")

    # Fix TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    blank_mask = df["TotalCharges"].isna()
    df.loc[blank_mask, "TotalCharges"] = (
        df.loc[blank_mask, "tenure"] * df.loc[blank_mask, "MonthlyCharges"]
    )

    # Strip whitespace from object columns
    obj_cols = df.select_dtypes(include="object").columns
    df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())

    return df


# ---------------------------------------------------------------------------
# Target encoding
# ---------------------------------------------------------------------------

def encode_target(df: pd.DataFrame) -> tuple:
    """
    Separate features and target.
    Returns (X: DataFrame, y: Series[int]) where y = 1 for churn.
    """
    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["Churn"])
    return X, y


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    """
    Build a ColumnTransformer with three sub-pipelines:
      1. Numeric: median imputation + StandardScaler
      2. Binary categorical: mode imputation + OrdinalEncoder (0/1)
      3. Multi-class categorical: mode imputation + OneHotEncoder (drop first)
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    # Include SeniorCitizen (already int) in numeric features
    numeric_cols = NUMERIC_FEATURES + ["SeniorCitizen"]

    binary_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    multi_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            drop="first",
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num",    numeric_pipeline,  numeric_cols),
            ("binary", binary_pipeline,   CATEGORICAL_BINARY),
            ("multi",  multi_pipeline,    CATEGORICAL_MULTI),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


# ---------------------------------------------------------------------------
# Feature names extraction
# ---------------------------------------------------------------------------

def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """
    Extract human-readable feature names from a fitted ColumnTransformer.
    Must be called after preprocessor.fit().
    """
    names = []

    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(transformer, "get_feature_names_out"):
            step_names = list(transformer.get_feature_names_out())
        else:
            # Pipeline: get names from the last step
            last_step = transformer.steps[-1][1]
            if hasattr(last_step, "get_feature_names_out"):
                # Pass input features for OHE
                step_names = list(last_step.get_feature_names_out())
            else:
                step_names = list(cols) if not isinstance(cols, str) else [cols]
        names.extend(step_names)

    return names


# ---------------------------------------------------------------------------
# Full prepare pipeline
# ---------------------------------------------------------------------------

def prepare_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Full data preparation pipeline:
      1. clean_raw_data
      2. encode_target -> X, y
      3. Stratified train/test split
      4. Build and fit ColumnTransformer on X_train only
      5. Transform both splits
      6. Extract feature names

    Returns:
      (X_train_proc, X_test_proc, y_train, y_test, preprocessor, feature_names)
      X arrays are np.ndarray; y are pd.Series; feature_names is list[str].
    """
    df = clean_raw_data(df)
    X, y = encode_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc  = preprocessor.transform(X_test)

    feature_names = get_feature_names(preprocessor)

    print(f"  Train shape: {X_train_proc.shape}  |  Test shape: {X_test_proc.shape}")
    print(f"  Features ({len(feature_names)}): {feature_names[:6]} ...")

    return X_train_proc, X_test_proc, y_train, y_test, preprocessor, feature_names
