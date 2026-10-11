"""Clean the raw Telco churn file and save data/processed/churn_clean.csv.

Cleaning rules (explained in reports/data_quality.md):
1. Column names are converted to snake_case.
2. tenure, MonthlyCharges and TotalCharges become numbers.
3. The 11 blank TotalCharges values belong to customers with tenure 0 (not billed yet),
   so they are set to 0 and flagged with is_new_customer.
4. SeniorCitizen (0/1) becomes Yes/No like the other yes/no columns.
5. 'No internet service' and 'No phone service' are kept as their own values,
   because they carry information.
6. No rows are removed.

Columns added for analysis:
    churned              1 if Churn is Yes, else 0
    is_new_customer      1 if tenure is 0 months
    tenure_group         0-12, 13-24, 25-48 or 49-72 months of tenure
    addon_services       number of internet add-ons the customer has (0-6)
"""

import pandas as pd

from config import CLEAN_FILE, EXPECTED_ROWS, PROCESSED_DIR, RAW_FILE, TABLES_DIR

OUT_DIR = TABLES_DIR / "cleaning"

COLUMN_NAMES = {
    "customerID": "customer_id",
    "gender": "gender",
    "SeniorCitizen": "senior_citizen",
    "Partner": "partner",
    "Dependents": "dependents",
    "tenure": "tenure_months",
    "PhoneService": "phone_service",
    "MultipleLines": "multiple_lines",
    "InternetService": "internet_service",
    "OnlineSecurity": "online_security",
    "OnlineBackup": "online_backup",
    "DeviceProtection": "device_protection",
    "TechSupport": "tech_support",
    "StreamingTV": "streaming_tv",
    "StreamingMovies": "streaming_movies",
    "Contract": "contract",
    "PaperlessBilling": "paperless_billing",
    "PaymentMethod": "payment_method",
    "MonthlyCharges": "monthly_charges",
    "TotalCharges": "total_charges",
    "Churn": "churn",
}

ADDON_COLUMNS = ["online_security", "online_backup", "device_protection",
                 "tech_support", "streaming_tv", "streaming_movies"]

# "months" in the label stops spreadsheet programs reading 0-12 as a date
TENURE_GROUPS = ["0-12 months", "13-24 months", "25-48 months", "49-72 months"]

YES_NO = {"Yes", "No"}
ADDON_VALUES = {"Yes", "No", "No internet service"}

# allowed values for every text column in the clean file
ALLOWED_VALUES = {
    "gender": {"Female", "Male"},
    "senior_citizen": YES_NO,
    "partner": YES_NO,
    "dependents": YES_NO,
    "phone_service": YES_NO,
    "multiple_lines": {"Yes", "No", "No phone service"},
    "internet_service": {"DSL", "Fiber optic", "No"},
    **{column: ADDON_VALUES for column in ADDON_COLUMNS},
    "contract": {"Month-to-month", "One year", "Two year"},
    "paperless_billing": YES_NO,
    "payment_method": {"Bank transfer (automatic)", "Credit card (automatic)",
                       "Electronic check", "Mailed check"},
    "churn": YES_NO,
    "tenure_group": set(TENURE_GROUPS),
}


class CheckFailed(Exception):
    pass


def clean(raw):
    df = raw.rename(columns=COLUMN_NAMES)

    df["tenure_months"] = pd.to_numeric(df["tenure_months"]).astype(int)
    df["monthly_charges"] = pd.to_numeric(df["monthly_charges"])

    # blank total charges only happen for customers who have not been billed yet
    df["is_new_customer"] = (df["tenure_months"] == 0).astype(int)
    df["total_charges"] = pd.to_numeric(df["total_charges"].str.strip().replace("", "0"))

    df["senior_citizen"] = df["senior_citizen"].map({"0": "No", "1": "Yes"})

    df["churned"] = (df["churn"] == "Yes").astype(int)
    df["tenure_group"] = pd.cut(df["tenure_months"], bins=[-1, 12, 24, 48, 72],
                                labels=TENURE_GROUPS).astype(str)
    df["addon_services"] = df[ADDON_COLUMNS].eq("Yes").sum(axis=1)
    return df


def validate(raw, df):
    """Return a table of checks. Every check must pass before the file is saved."""
    no_internet = df["internet_service"] == "No"
    checks = [
        ("row count equals raw file", len(df) == len(raw)),
        (f"row count is {EXPECTED_ROWS:,}", len(df) == EXPECTED_ROWS),
        ("customer_id is unique", df["customer_id"].is_unique),
        ("no missing values in any column", not df.isna().any().any()),
        ("tenure_months between 0 and 72", df["tenure_months"].between(0, 72).all()),
        ("monthly_charges above 0", (df["monthly_charges"] > 0).all()),
        ("total_charges is 0 only for new customers",
         ((df["total_charges"] == 0) == (df["is_new_customer"] == 1)).all()),
        ("churned matches churn", ((df["churned"] == 1) == (df["churn"] == "Yes")).all()),
        ("churn counts unchanged from raw",
         df["churned"].sum() == (raw["Churn"] == "Yes").sum()),
        ("sum of monthly_charges unchanged from raw",
         abs(df["monthly_charges"].sum() - pd.to_numeric(raw["MonthlyCharges"]).sum()) < 0.01),
        ("customers without internet have 0 add-ons", (df.loc[no_internet, "addon_services"] == 0).all()),
        ("addon_services between 0 and 6", df["addon_services"].between(0, 6).all()),
    ]
    for column, allowed in ALLOWED_VALUES.items():
        checks.append((f"{column} has only allowed values", set(df[column].unique()) <= allowed))

    return pd.DataFrame(checks, columns=["check", "passed"]).astype({"passed": bool})


def cleaning_summary(raw, df):
    return pd.DataFrame([
        {"item": "rows in raw file", "value": len(raw)},
        {"item": "rows in clean file", "value": len(df)},
        {"item": "columns in raw file", "value": raw.shape[1]},
        {"item": "columns in clean file", "value": df.shape[1]},
        {"item": "blank total_charges set to 0", "value": int((raw["TotalCharges"].str.strip() == "").sum())},
        {"item": "new customers (tenure 0)", "value": int(df["is_new_customer"].sum())},
        {"item": "customers who churned", "value": int(df["churned"].sum())},
        {"item": "customers without internet service", "value": int((df["internet_service"] == "No").sum())},
        {"item": "customers without phone service", "value": int((df["phone_service"] == "No").sum())},
    ])


def clean_data(verbose=True):
    raw = pd.read_csv(RAW_FILE, dtype=str, keep_default_na=False)
    df = clean(raw)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    checks = validate(raw, df)
    checks.to_csv(OUT_DIR / "validation_checks.csv", index=False)
    failed = checks[~checks["passed"]]
    if not failed.empty:
        raise CheckFailed("Cleaning checks failed:\n" + failed.to_string(index=False))

    summary = cleaning_summary(raw, df)
    summary.to_csv(OUT_DIR / "cleaning_summary.csv", index=False)
    tenure_groups = (df.groupby("tenure_group")
                     .agg(customers=("customer_id", "count"),
                          min_tenure=("tenure_months", "min"),
                          max_tenure=("tenure_months", "max"))
                     .reset_index())
    tenure_groups.to_csv(OUT_DIR / "tenure_groups.csv", index=False)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_FILE, index=False)

    if verbose:
        print(summary.to_string(index=False))
        print(f"\n{tenure_groups.to_string(index=False)}")
    print(f"Check passed: all {len(checks)} cleaning checks.")
    print(f"Saved {CLEAN_FILE.name}: {len(df):,} rows, {df.shape[1]} columns")
    return df


if __name__ == "__main__":
    clean_data()
