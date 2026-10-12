"""Exploratory analysis of churn. Writes CSV tables to reports/tables/analysis/.

Definitions (see docs/analysis_notes.md):
    churn rate            churned customers / customers in the group
    monthly revenue lost  sum of monthly_charges of churned customers
"""

import pandas as pd

from config import CLEAN_FILE, TABLES_DIR

OUT_DIR = TABLES_DIR / "analysis"

CATEGORICAL_FEATURES = [
    "contract", "tenure_group", "internet_service", "payment_method",
    "paperless_billing", "senior_citizen", "partner", "dependents",
    "phone_service", "multiple_lines", "online_security", "online_backup",
    "device_protection", "tech_support", "streaming_tv", "streaming_movies",
    "gender", "addon_services",
]
NUMERIC_FEATURES = ["tenure_months", "monthly_charges", "total_charges"]

MIN_CELL = 50       # two-way cells below this get no churn rate and are flagged
MIN_SEGMENT = 100   # segments below this are left out of the risk ranking
SEGMENT_PAIRS = [("contract", "internet_service"), ("contract", "tenure_group"),
                 ("internet_service", "payment_method")]


def load_clean():
    return pd.read_csv(CLEAN_FILE)


def overall_summary(df):
    """Customers, churn, monthly charges and the monthly charges of churners."""
    churned = df[df["churned"] == 1]
    total_monthly = df["monthly_charges"].sum()
    lost = churned["monthly_charges"].sum()
    rows = [
        ("customers", len(df)),
        ("churned_customers", len(churned)),
        ("churn_rate_pct", round(100 * len(churned) / len(df), 2)),
        ("total_monthly_charges", round(total_monthly, 2)),
        ("monthly_charges_of_churned", round(lost, 2)),
        ("share_of_monthly_charges_lost_pct", round(100 * lost / total_monthly, 2)),
        ("avg_monthly_charges_churned", round(churned["monthly_charges"].mean(), 2)),
        ("avg_monthly_charges_stayed",
         round(df.loc[df["churned"] == 0, "monthly_charges"].mean(), 2)),
    ]
    return pd.DataFrame(rows, columns=["measure", "value"])


def group_summary(df, by):
    """Customers, churners, churn rate and shares of churners and monthly charges."""
    df = df.assign(churned_charges=df["monthly_charges"] * df["churned"])
    g = df.groupby(by, observed=True).agg(
        customers=("churned", "size"),
        churned=("churned", "sum"),
        monthly_charges=("monthly_charges", "sum"),
        churned_monthly_charges=("churned_charges", "sum"),
    ).reset_index()
    g["churn_rate_pct"] = (100 * g["churned"] / g["customers"]).round(2)
    g["share_of_customers_pct"] = (100 * g["customers"] / len(df)).round(2)
    g["share_of_churners_pct"] = (100 * g["churned"] / df["churned"].sum()).round(2)
    g["share_of_monthly_charges_pct"] = (
        100 * g["monthly_charges"] / df["monthly_charges"].sum()).round(2)
    g["monthly_charges"] = g["monthly_charges"].round(2)
    g["churned_monthly_charges"] = g["churned_monthly_charges"].round(2)
    return g


def churn_by_feature(df):
    """Long table: one row per feature value, for every categorical feature."""
    parts = []
    for feature in CATEGORICAL_FEATURES:
        g = group_summary(df, feature).rename(columns={feature: "value"})
        g["value"] = g["value"].astype(str)
        g.insert(0, "feature", feature)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def feature_gaps(long_table):
    """Per feature: highest and lowest churn group and the gap between them."""
    rows = []
    for feature, g in long_table.groupby("feature", sort=False):
        top = g.loc[g["churn_rate_pct"].idxmax()]
        low = g.loc[g["churn_rate_pct"].idxmin()]
        rows.append({
            "feature": feature,
            "highest_group": top["value"], "highest_rate_pct": top["churn_rate_pct"],
            "highest_n": top["customers"],
            "lowest_group": low["value"], "lowest_rate_pct": low["churn_rate_pct"],
            "lowest_n": low["customers"],
            "gap_pct_points": round(top["churn_rate_pct"] - low["churn_rate_pct"], 2),
        })
    out = pd.DataFrame(rows).sort_values("gap_pct_points", ascending=False)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out.reset_index(drop=True)


def numeric_comparison(df):
    """Mean and median of numeric features for churned vs stayed customers."""
    rows = []
    for col in NUMERIC_FEATURES:
        for label, flag in (("churned", 1), ("stayed", 0)):
            s = df.loc[df["churned"] == flag, col]
            rows.append({"feature": col, "group": label, "customers": len(s),
                         "mean": round(s.mean(), 2), "median": round(s.median(), 2)})
    return pd.DataFrame(rows)


def two_way(df, row_feature, col_feature):
    """Churn rate for every row x column combination, with n for every cell.

    Cells with fewer than MIN_CELL customers keep their n but get no churn rate
    and are flagged, because a rate from so few customers is unreliable.
    """
    g = group_summary(df, [row_feature, col_feature])
    g["small_cell"] = g["customers"] < MIN_CELL
    g.loc[g["small_cell"], "churn_rate_pct"] = float("nan")
    keep = [row_feature, col_feature, "customers", "churned", "churn_rate_pct",
            "small_cell"]
    return g[keep]


def highest_risk_segments(df):
    """Segments with at least MIN_SEGMENT customers, ranked by churn rate.

    A segment is one feature value (e.g. contract = Month-to-month) or a pair
    of two features (e.g. contract and internet_service).
    """
    keep = ["segment", "customers", "churned", "churn_rate_pct",
            "share_of_churners_pct", "share_of_monthly_charges_pct"]
    singles = churn_by_feature(df)
    singles["segment"] = singles["feature"] + " = " + singles["value"]
    parts = [singles[keep]]
    for a, b in SEGMENT_PAIRS:
        g = group_summary(df, [a, b])
        g["segment"] = (a + " = " + g[a].astype(str) + " & "
                        + b + " = " + g[b].astype(str))
        parts.append(g[keep])
    allseg = pd.concat(parts, ignore_index=True)
    allseg = allseg[allseg["customers"] >= MIN_SEGMENT]
    allseg = allseg.sort_values(["churn_rate_pct", "customers"], ascending=False)
    allseg.insert(0, "rank", range(1, len(allseg) + 1))
    return allseg.reset_index(drop=True)


def check_tables(df, tables):
    """Stop with an error if the tables do not add up to the clean data."""
    total_churn = int(df["churned"].sum())
    for feature, g in tables["churn_by_feature"].groupby("feature"):
        if g["churned"].sum() != total_churn:
            raise ValueError(f"{feature}: churn counts do not sum to {total_churn}")
        if g["customers"].sum() != len(df):
            raise ValueError(f"{feature}: customers do not sum to {len(df)}")
    overall = tables["overall_summary"].set_index("measure")["value"]
    if overall["churned_customers"] != total_churn:
        raise ValueError("overall summary: wrong churn count")
    for name in ("segments_contract_internet", "segments_contract_tenure"):
        t = tables[name]
        if t["customers"].sum() != len(df) or t["churned"].sum() != total_churn:
            raise ValueError(f"{name}: counts do not add up to the clean data")
    risk = tables["highest_risk_segments"]
    if (risk["customers"] < MIN_SEGMENT).any():
        raise ValueError("highest_risk_segments: segment below the minimum size")


def run_exploration(verbose=True):
    """Build all analysis tables, check them, and save them."""
    df = load_clean()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tables = {
        "overall_summary": overall_summary(df),
        "churn_by_feature": churn_by_feature(df),
        "numeric_comparison": numeric_comparison(df),
        "segments_contract_internet": two_way(df, "contract", "internet_service"),
        "segments_contract_tenure": two_way(df, "contract", "tenure_group"),
        "highest_risk_segments": highest_risk_segments(df),
    }
    tables["feature_gaps"] = feature_gaps(tables["churn_by_feature"])

    check_tables(df, tables)

    for name, table in tables.items():
        table.to_csv(OUT_DIR / f"{name}.csv", index=False)
        if verbose:
            print(f"\n{name}\n{table.head(10).to_string(index=False)}")
    return tables


if __name__ == "__main__":
    run_exploration()
