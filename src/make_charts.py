"""Make the report charts (reports/figures/*.png) from the analysis tables.

The data behind each chart is saved to reports/tables/figures/.
Colours: grey for ordinary groups, one red accent for the highest-churn group.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import FIGURES_DIR, TABLES_DIR  # noqa: E402
from explore import group_summary, load_clean, two_way  # noqa: E402
from stat_tests import run_tests  # noqa: E402

DATA_DIR = TABLES_DIR / "figures"
GREY = "#9AA5B1"
ACCENT = "#D62728"
DARK = "#1F2933"
BLUE = "#2F6690"
DPI = 150

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#52606D", "axes.titlelocation": "left",
    "figure.dpi": 100, "savefig.dpi": DPI, "font.family": "DejaVu Sans",
})


def save(fig, name, data):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=DPI, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    data.to_csv(DATA_DIR / f"{name}.csv", index=False)


def rate_bars(df, feature, order, title, xlabel, name, overall):
    """Vertical bars of churn rate with n shown under each label."""
    g = group_summary(df, feature).set_index(feature).loc[order].reset_index()
    colors = [ACCENT if r == g["churn_rate_pct"].max() else GREY
              for r in g["churn_rate_pct"]]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    bars = ax.bar(range(len(g)), g["churn_rate_pct"], color=colors, width=0.62)
    for bar, rate, n in zip(bars, g["churn_rate_pct"], g["customers"]):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 1, f"{rate:.1f}%",
                ha="center", va="bottom", fontweight="bold", color=DARK)
    ax.axhline(overall, color=DARK, linestyle="--", linewidth=1)
    ax.text(len(g) - 0.5, overall + 1, f"All customers: {overall:.1f}%", ha="right",
            va="bottom", fontsize=9.5, color=DARK)
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels([f"{v}\n(n = {n:,})" for v, n in zip(g[feature], g["customers"])])
    ax.set_ylim(0, max(g["churn_rate_pct"]) * 1.18)
    ax.set_ylabel("Churn rate (% of customers who left)")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    save(fig, name, g[[feature, "customers", "churned", "churn_rate_pct"]])


def chart_contract(df, overall):
    rate_bars(df, "contract", ["Month-to-month", "One year", "Two year"],
              "Month-to-month customers churn far more\nthan customers on longer contracts",
              "Contract type", "01_churn_by_contract", overall)


def chart_tenure(df, overall):
    rate_bars(df, "tenure_group",
              ["0-12 months", "13-24 months", "25-48 months", "49-72 months"],
              "Churn is highest in the first year\nand falls steadily with tenure",
              "Tenure (months as a customer)", "02_churn_by_tenure", overall)


def chart_internet_payment(df, overall):
    """Two horizontal panels sharing a scale: internet service and payment method."""
    parts = []
    for feature in ("internet_service", "payment_method"):
        g = group_summary(df, feature).rename(columns={feature: "value"})
        g.insert(0, "feature", feature)
        parts.append(g.sort_values("churn_rate_pct"))
    data = pd.concat(parts)
    top = data["churn_rate_pct"].max()
    fig, axes = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True,
                             gridspec_kw={"height_ratios": [3, 4]})
    titles = {"internet_service": "Internet service", "payment_method": "Payment method"}
    for ax, (feature, g) in zip(axes, data.groupby("feature", sort=False)):
        colors = [ACCENT if r == g["churn_rate_pct"].max() else GREY
                  for r in g["churn_rate_pct"]]
        ax.barh(range(len(g)), g["churn_rate_pct"], color=colors, height=0.6)
        for i, (rate, n) in enumerate(zip(g["churn_rate_pct"], g["customers"])):
            ax.text(rate + 0.8, i, f"{rate:.1f}%  (n = {n:,})", va="center",
                    fontsize=10, color=DARK,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
        ax.set_yticks(range(len(g)))
        ax.set_yticklabels(g["value"])
        ax.axvline(overall, color=DARK, linestyle="--", linewidth=1)
        ax.set_title(titles[feature], fontsize=11.5)
        ax.set_xlim(0, top * 1.4)
    axes[1].set_xlabel("Churn rate (% of customers who left); dashed line = all customers")
    fig.suptitle("Fiber optic and electronic-check customers churn the most",
                 x=0.125, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "03_churn_by_internet_and_payment", data[
        ["feature", "value", "customers", "churned", "churn_rate_pct"]])


def chart_monthly_charges(df):
    bins = np.arange(15, 125, 5)
    fig, ax = plt.subplots(figsize=(8, 5))
    rows = []
    for flag, label, color in ((0, "Stayed", GREY), (1, "Churned", ACCENT)):
        s = df.loc[df["churned"] == flag, "monthly_charges"]
        ax.hist(s, bins=bins, density=True, alpha=0.65, color=color,
                label=f"{label} (n = {len(s):,}, median ${s.median():.2f})")
        ax.axvline(s.median(), color=color, linewidth=2)
        counts, _ = np.histogram(s, bins=bins)
        rows.append(pd.DataFrame({"group": label, "bin_start": bins[:-1],
                                  "bin_end": bins[1:], "customers": counts}))
    ax.set_xlabel("Monthly charges (US dollars)")
    ax.set_ylabel("Share of customers in group (density)")
    ax.set_title("Churned customers pay more per month than customers who stayed")
    ax.set_ylim(0, 0.032)
    ax.legend(loc="upper right", facecolor="white", edgecolor="none", framealpha=1)
    save(fig, "04_monthly_charges_distribution", pd.concat(rows, ignore_index=True))


def chart_heatmap(df):
    data = two_way(df, "contract", "internet_service")
    rows = ["Month-to-month", "One year", "Two year"]
    cols = ["DSL", "Fiber optic", "No"]
    rate = data.pivot(index="contract", columns="internet_service",
                      values="churn_rate_pct").loc[rows, cols]
    n = data.pivot(index="contract", columns="internet_service",
                   values="customers").loc[rows, cols]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    im = ax.imshow(rate.values, cmap="Reds", vmin=0, vmax=60)
    for i in range(len(rows)):
        for j in range(len(cols)):
            r = rate.values[i, j]
            ax.text(j, i, f"{r:.1f}%\nn = {int(n.values[i, j]):,}", ha="center",
                    va="center", color="white" if r > 35 else DARK, fontsize=11)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(["DSL", "Fiber optic", "No internet"])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_xlabel("Internet service")
    ax.set_ylabel("Contract type")
    ax.set_title("Month-to-month fiber optic customers churn most (54.6%)")
    ax.spines[:].set_visible(False)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Churn rate (%)")
    save(fig, "05_churn_heatmap_contract_internet", data)


def chart_feature_ranking(tests):
    cat = tests[tests["test"] == "chi-square"].sort_values("effect_size")
    labels = [f.replace("_", " ") for f in cat["feature"]]
    colors = [ACCENT if v >= 0.3 else (BLUE if v >= 0.1 else GREY)
              for v in cat["effect_size"]]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(range(len(cat)), cat["effect_size"], color=colors, height=0.65)
    for i, v in enumerate(cat["effect_size"]):
        ax.text(v + 0.006, i, f"{v:.2f}", va="center", fontsize=9.5, color=DARK)
    ax.set_yticks(range(len(cat)))
    ax.set_yticklabels(labels)
    ax.set_xlim(0, cat["effect_size"].max() * 1.15)
    ax.set_xlabel("Association with churn (Cramér's V, 0 = none, 1 = perfect)")
    ax.set_title("Contract, tenure and support services show the strongest\n"
                 "association with churn; gender and phone service almost none")
    ax.text(0.99, 0.03, f"n = {int(tests['n'].iloc[0]):,} customers\n"
            "Red: V ≥ 0.30   Blue: 0.10 – 0.30   Grey: < 0.10",
            transform=ax.transAxes, ha="right", fontsize=9, color=DARK)
    save(fig, "06_feature_association_ranking",
         cat[["feature", "effect_size", "effect_label", "p_value", "p_holm"]]
         .sort_values("effect_size", ascending=False))


def make_charts():
    df = load_clean()
    overall = 100 * df["churned"].mean()
    tests = run_tests(df)
    chart_contract(df, overall)
    chart_tenure(df, overall)
    chart_internet_payment(df, overall)
    chart_monthly_charges(df)
    chart_heatmap(df)
    chart_feature_ranking(tests)
    made = sorted(p.name for p in FIGURES_DIR.glob("*.png"))
    if len(made) != 6:
        raise ValueError(f"expected 6 charts, found {len(made)}")
    return made


if __name__ == "__main__":
    for name in make_charts():
        print("Saved", name)
