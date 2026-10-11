"""Shared paths and constants for the project."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
RAW_FILE = RAW_DIR / "Telco-Customer-Churn.csv"
CLEAN_FILE = PROCESSED_DIR / "churn_clean.csv"

TABLES_DIR = PROJECT_DIR / "reports" / "tables"
FIGURES_DIR = PROJECT_DIR / "reports" / "figures"

# Public copy of the dataset published by IBM
DATA_URL = ("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
            "master/data/Telco-Customer-Churn.csv")
# SHA-256 of the file this project was built with
DATA_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"

EXPECTED_ROWS = 7043
