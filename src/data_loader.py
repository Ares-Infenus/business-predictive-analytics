"""
data_loader.py
--------------
Download, load, and validate the IBM Telco Customer Churn dataset.
Falls back to synthetic data generation if download fails.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

RAW_DATA_PATH = Path("data/raw/Telco-Customer-Churn.csv")

EXPECTED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
]


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_data(url: str = DATA_URL, save_path: Path = RAW_DATA_PATH,
                  timeout: int = 30) -> bool:
    """
    Download the CSV from `url` and save to `save_path`.
    Returns True on success, False on any network/IO error.
    """
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        headers = {"User-Agent": "Mozilla/5.0"}
        print(f"  Downloading from: {url}")
        response = requests.get(url, headers=headers, timeout=timeout, stream=True)
        response.raise_for_status()

        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"  Saved to: {save_path}")
        return True

    except Exception as exc:
        print(f"  Download failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Synthetic fallback
# ---------------------------------------------------------------------------

def generate_synthetic_data(n_samples: int = 7043, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic DataFrame that closely mimics the Telco schema and
    real-world distributions. Used as fallback when download is unavailable.
    """
    rng = np.random.default_rng(random_state)

    gender         = rng.choice(["Male", "Female"], size=n_samples)
    senior         = rng.choice([0, 1], size=n_samples, p=[0.84, 0.16])
    partner        = rng.choice(["Yes", "No"], size=n_samples, p=[0.48, 0.52])
    dependents     = rng.choice(["Yes", "No"], size=n_samples, p=[0.30, 0.70])
    tenure         = rng.integers(0, 73, size=n_samples)
    phone_service  = rng.choice(["Yes", "No"], size=n_samples, p=[0.90, 0.10])
    multiple_lines = rng.choice(
        ["No", "Yes", "No phone service"], size=n_samples, p=[0.42, 0.42, 0.16]
    )
    internet       = rng.choice(
        ["DSL", "Fiber optic", "No"], size=n_samples, p=[0.34, 0.44, 0.22]
    )

    def internet_dependent(yes_p=0.28):
        return [
            "No internet service" if internet[i] == "No"
            else ("Yes" if rng.random() < yes_p else "No")
            for i in range(n_samples)
        ]

    online_sec      = internet_dependent(0.29)
    online_backup   = internet_dependent(0.34)
    device_prot     = internet_dependent(0.34)
    tech_support    = internet_dependent(0.29)
    streaming_tv    = internet_dependent(0.38)
    streaming_mov   = internet_dependent(0.39)

    contract        = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n_samples, p=[0.55, 0.24, 0.21]
    )
    paperless       = rng.choice(["Yes", "No"], size=n_samples, p=[0.59, 0.41])
    payment         = rng.choice(
        ["Electronic check", "Mailed check",
         "Bank transfer (automatic)", "Credit card (automatic)"],
        size=n_samples, p=[0.34, 0.23, 0.22, 0.21]
    )

    monthly_charges = rng.uniform(18.25, 118.75, size=n_samples).round(2)
    total_charges   = (tenure * monthly_charges + rng.normal(0, 10, size=n_samples)).clip(0)
    total_charges_str = [
        " " if tenure[i] == 0 else str(round(total_charges[i], 2))
        for i in range(n_samples)
    ]

    # Churn probability driven by contract + tenure
    churn_logit = (
        -1.5
        + (contract == "Month-to-month") * 1.2
        + (internet == "Fiber optic") * 0.5
        + (tenure < 6) * 0.8
        - tenure * 0.02
        + (online_sec == "No") * 0.3
        + rng.normal(0, 0.5, size=n_samples)
    )
    churn_prob  = 1 / (1 + np.exp(-churn_logit))
    churn       = ["Yes" if p > 0.5 else "No" for p in churn_prob]

    customer_ids = [f"SYN-{i:07d}" for i in range(n_samples)]

    df = pd.DataFrame({
        "customerID":      customer_ids,
        "gender":          gender,
        "SeniorCitizen":   senior,
        "Partner":         partner,
        "Dependents":      dependents,
        "tenure":          tenure,
        "PhoneService":    phone_service,
        "MultipleLines":   multiple_lines,
        "InternetService": internet,
        "OnlineSecurity":  online_sec,
        "OnlineBackup":    online_backup,
        "DeviceProtection":device_prot,
        "TechSupport":     tech_support,
        "StreamingTV":     streaming_tv,
        "StreamingMovies": streaming_mov,
        "Contract":        contract,
        "PaperlessBilling":paperless,
        "PaymentMethod":   payment,
        "MonthlyCharges":  monthly_charges,
        "TotalCharges":    total_charges_str,
        "Churn":           churn,
    })

    return df


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_data(force_download: bool = False) -> pd.DataFrame:
    """
    Load the Telco churn dataset.

    Priority:
      1. Return cached CSV if it exists and force_download is False.
      2. Download from GitHub URL.
      3. Fall back to synthetic generation.
    """
    if RAW_DATA_PATH.exists() and not force_download:
        print(f"  Loading cached data from {RAW_DATA_PATH}")
        return pd.read_csv(RAW_DATA_PATH)

    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    success = download_data()

    if not success:
        warnings.warn(
            "Could not download Telco dataset. Generating synthetic data. "
            "Results may differ from the real dataset.",
            UserWarning,
            stacklevel=2,
        )
        df = generate_synthetic_data()
        df.to_csv(RAW_DATA_PATH, index=False)
        return df

    return pd.read_csv(RAW_DATA_PATH)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_data(df: pd.DataFrame) -> dict:
    """
    Run schema and quality checks. Returns a validation report dict.
    Raises ValueError if expected columns are missing.
    """
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")

    churn_rate = (df["Churn"] == "Yes").mean()

    total_charges_blanks = (
        df["TotalCharges"].astype(str).str.strip() == ""
    ).sum()

    missing_values = {
        col: int(df[col].isna().sum())
        for col in df.columns
        if df[col].isna().sum() > 0
    }

    report = {
        "columns_present":      True,
        "row_count":            len(df),
        "churn_rate":           float(churn_rate),
        "missing_values":       missing_values,
        "totalcharges_blanks":  int(total_charges_blanks),
        "duplicate_customer_ids": int(df["customerID"].duplicated().sum()),
    }

    return report
