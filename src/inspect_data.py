"""Inspect the raw file before any cleaning.

Every column is read as text so nothing is changed or guessed by pandas.
Results are saved to reports/tables/inspection/.
"""

import pandas as pd

from config import RAW_FILE, TABLES_DIR

OUT_DIR = TABLES_DIR / "inspection"

INTERNET_ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                   "TechSupport", "StreamingTV", "StreamingMovies"]


def read_raw():
    return pd.read_csv(RAW_FILE, dtype=str, keep_default_na=False)


def column_profile(df):
    """One row per column: distinct values, blanks and the values themselves if few."""
    rows = []
    for column in df.columns:
        values = df[column]
        distinct = values.nunique()
        rows.append({
            "column": column,
            "distinct_values": distinct,
            "blank": int((values.str.strip() == "").sum()),
            "has_extra_spaces": int((values != values.str.strip()).sum()),
            "values": " | ".join(sorted(values.unique())) if distinct <= 6 else "",
        })
    return pd.DataFrame(rows)


def numeric_summary(df):
    """Summary of the three numeric columns (blank values excluded)."""
    numeric = df[["tenure", "MonthlyCharges", "TotalCharges"]].apply(pd.to_numeric, errors="coerce")
    summary = numeric.describe().T.round(2)
    summary.insert(0, "not_numeric", numeric.isna().sum())
    return summary.rename_axis("column").reset_index()


def consistency_checks(df):
    """Rules the data should follow. 'rows_failing' should be 0 for each."""
    no_internet = df["InternetService"] == "No"
    addons_say_no_internet = df[INTERNET_ADDONS].eq("No internet service")
    no_phone = df["PhoneService"] == "No"
    lines_say_no_phone = df["MultipleLines"] == "No phone service"
    blank_total = df["TotalCharges"].str.strip() == ""

    checks = {
        "duplicate customerID": df["customerID"].duplicated().sum(),
        "duplicate full rows": df.duplicated().sum(),
        "no internet but an add-on is not 'No internet service'":
            (no_internet & ~addons_say_no_internet.all(axis=1)).sum(),
        "has internet but an add-on says 'No internet service'":
            (~no_internet & addons_say_no_internet.any(axis=1)).sum(),
        "no phone but MultipleLines is not 'No phone service'": (no_phone & ~lines_say_no_phone).sum(),
        "has phone but MultipleLines says 'No phone service'": (~no_phone & lines_say_no_phone).sum(),
        "blank TotalCharges with tenure above 0": (blank_total & (df["tenure"] != "0")).sum(),
        "tenure 0 with a TotalCharges value": (~blank_total & (df["tenure"] == "0")).sum(),
        "customer with neither phone nor internet": (no_phone & no_internet).sum(),
    }
    return pd.DataFrame({"check": checks.keys(), "rows_failing": [int(v) for v in checks.values()]})


def churn_distribution(df):
    counts = df["Churn"].value_counts().rename_axis("Churn").reset_index(name="customers")
    counts["share_pct"] = (100 * counts["customers"] / len(df)).round(2)
    return counts


def total_vs_tenure_times_monthly(df):
    """TotalCharges compared with tenure x MonthlyCharges (they are not expected to be equal)."""
    numeric = df[["tenure", "MonthlyCharges", "TotalCharges"]].apply(pd.to_numeric, errors="coerce").dropna()
    estimate = numeric["tenure"] * numeric["MonthlyCharges"]
    pct_diff = 100 * (numeric["TotalCharges"] - estimate).abs() / numeric["TotalCharges"]
    return pd.DataFrame([{
        "customers_compared": len(numeric),
        "median_abs_diff_pct": round(pct_diff.median(), 2),
        "p90_abs_diff_pct": round(pct_diff.quantile(0.90), 2),
        "max_abs_diff_pct": round(pct_diff.max(), 2),
        "within_10_pct": int((pct_diff <= 10).sum()),
    }])


def inspect_data(verbose=True):
    df = read_raw()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "shape": pd.DataFrame([{"rows": len(df), "columns": df.shape[1]}]),
        "column_profile": column_profile(df),
        "numeric_summary": numeric_summary(df),
        "consistency_checks": consistency_checks(df),
        "churn_distribution": churn_distribution(df),
        "blank_total_charges": df.loc[df["TotalCharges"].str.strip() == "",
                                      ["customerID", "tenure", "Contract", "MonthlyCharges",
                                       "TotalCharges", "Churn"]],
        "total_vs_tenure_times_monthly": total_vs_tenure_times_monthly(df),
    }
    for name, table in results.items():
        table.to_csv(OUT_DIR / f"{name}.csv", index=False)
        if verbose:
            print(f"\n== {name}\n{table.to_string(index=False)}")
    return results


if __name__ == "__main__":
    inspect_data()
