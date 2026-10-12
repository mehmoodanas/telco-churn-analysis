"""Statistical tests of each feature against churn.

Writes reports/tables/analysis/statistical_tests.csv.

Categorical features: chi-square test of independence, effect size Cramer's V.
Numeric features:     Mann-Whitney U test, effect size rank-biserial correlation
                      (positive = churned customers have higher values).

With 7,043 customers almost any difference gives a tiny p-value, so the effect
size is what tells us whether a difference matters. Because 21 tests are run,
p-values are also adjusted with the Holm method (and Bonferroni for comparison).
See docs/analysis_notes.md.
"""

import numpy as np
import pandas as pd
from scipy import stats

from config import TABLES_DIR
from explore import CATEGORICAL_FEATURES, NUMERIC_FEATURES, load_clean

OUT_FILE = TABLES_DIR / "analysis" / "statistical_tests.csv"
ALPHA = 0.05


def cramers_v(table):
    """Cramer's V from a contingency table (0 = no association, 1 = perfect)."""
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    k = min(table.shape) - 1
    return float(np.sqrt(chi2 / (n * k)))


def effect_label(v):
    """Rough reading of Cramer's V for a 2-column table (Cohen's guide)."""
    if v < 0.1:
        return "negligible"
    if v < 0.3:
        return "small"
    if v < 0.5:
        return "medium"
    return "large"


def chi_square_test(df, feature):
    table = pd.crosstab(df[feature], df["churned"])
    chi2, p, dof, expected = stats.chi2_contingency(table)
    return {
        "feature": feature, "test": "chi-square", "statistic": chi2, "df": dof,
        "p_value": p, "effect_size_name": "cramers_v",
        "effect_size": cramers_v(table),
        "min_expected_count": float(expected.min()),
    }


def mann_whitney_test(df, feature):
    churned = df.loc[df["churned"] == 1, feature]
    stayed = df.loc[df["churned"] == 0, feature]
    u, p = stats.mannwhitneyu(churned, stayed, alternative="two-sided")
    # rank-biserial = 2U / (n1 * n2) - 1, with U counted for the churned group
    rbc = 2 * u / (len(churned) * len(stayed)) - 1
    return {
        "feature": feature, "test": "mann-whitney-u", "statistic": u, "df": np.nan,
        "p_value": p, "effect_size_name": "rank_biserial", "effect_size": float(rbc),
        "min_expected_count": np.nan,
    }


def holm_adjust(p_values):
    """Holm step-down adjusted p-values."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def run_tests(df=None):
    if df is None:
        df = load_clean()
    rows = [chi_square_test(df, f) for f in CATEGORICAL_FEATURES]
    rows += [mann_whitney_test(df, f) for f in NUMERIC_FEATURES]
    out = pd.DataFrame(rows)

    m = len(out)
    out["p_holm"] = holm_adjust(out["p_value"])
    out["p_bonferroni"] = np.minimum(1.0, out["p_value"] * m)
    out["significant_after_holm"] = out["p_holm"] < ALPHA
    out["effect_size_abs"] = out["effect_size"].abs()
    out["effect_label"] = out["effect_size_abs"].map(effect_label)
    out.insert(2, "n", len(df))
    out = out.sort_values("effect_size_abs", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))

    check_tests(out)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_FILE, index=False)
    return out


def check_tests(out):
    """Stop with an error if the results look wrong."""
    expected = len(CATEGORICAL_FEATURES) + len(NUMERIC_FEATURES)
    if len(out) != expected:
        raise ValueError(f"expected {expected} tests, got {len(out)}")
    if not out["p_value"].between(0, 1).all():
        raise ValueError("p-values outside 0-1")
    if (out["p_holm"] < out["p_value"] - 1e-12).any():
        raise ValueError("Holm p-values smaller than raw p-values")
    if not out["effect_size_abs"].between(0, 1).all():
        raise ValueError("effect sizes outside 0-1")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    result = run_tests()
    print(result[["rank", "feature", "test", "p_value", "p_holm", "effect_size",
                  "effect_label", "significant_after_holm"]].to_string(index=False))
