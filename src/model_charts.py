"""Charts for the prediction models (reports/figures/07..10), drawn from the
tables that src/model.py saves in reports/tables/model/.
The data behind each chart is also saved to reports/tables/figures/.
"""

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import TABLES_DIR  # noqa: E402
from make_charts import ACCENT, BLUE, DARK, GREY, save  # noqa: E402  (also sets chart style)
from model import BASELINE, BOOST, LOGREG  # noqa: E402

MODEL_DIR = TABLES_DIR / "model"
STYLE = {BASELINE: (GREY, "--"), LOGREG: (BLUE, "-"), BOOST: (DARK, "-")}


def read(name):
    return pd.read_csv(MODEL_DIR / f"{name}.csv")


def chart_roc_pr():
    comp = read("model_comparison").set_index("model")
    roc, pr = read("roc_curve"), read("pr_curve")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for name, (color, line) in STYLE.items():
        short = name.split(" (")[0]
        r = roc[roc["model"] == name]
        axes[0].plot(r["false_positive_rate"], r["true_positive_rate"], color=color,
                     linestyle=line, linewidth=2,
                     label=f"{short} (AUC {comp.loc[name, 'test_roc_auc']:.3f})")
        p = pr[pr["model"] == name]
        axes[1].plot(p["recall"], p["precision"], color=color, linestyle=line, linewidth=2,
                     label=f"{short} (AP {comp.loc[name, 'test_pr_auc']:.3f})")
    axes[0].set_xlabel("False positive rate (stayed customers flagged)")
    axes[0].set_ylabel("True positive rate (churners caught)")
    axes[0].set_title("ROC curve")
    axes[1].set_xlabel("Recall (churners caught)")
    axes[1].set_ylabel("Precision (flagged who really churn)")
    axes[1].set_title("Precision-recall curve")
    axes[1].set_ylim(0, 1.02)
    for ax in axes:
        ax.legend(loc="lower right" if ax is axes[0] else "upper right",
                  facecolor="white", edgecolor="none", framealpha=1)
    fig.suptitle("Both models rank churners well above chance on the held-out test set",
                 x=0.06, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    data = pd.concat([roc.assign(curve="roc"), pr.assign(curve="precision_recall")])
    save(fig, "07_roc_and_pr_curves", data)


def chart_threshold():
    t = read("threshold_analysis_test")
    chosen = t[t["chosen"] == 1].iloc[0]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(t["threshold"], 100 * t["recall"], color=ACCENT, linewidth=2.2,
            label="Recall (share of churners caught)")
    ax.plot(t["threshold"], 100 * t["precision"], color=BLUE, linewidth=2.2,
            label="Precision (share of flagged who churn)")
    ax.plot(t["threshold"], t["flagged_pct_of_customers"], color=GREY, linewidth=2,
            linestyle="--", label="Customers flagged (% of all)")
    ax.axvline(chosen["threshold"], color=DARK, linewidth=1)
    ax.annotate(
        f"Chosen threshold {chosen['threshold']:.2f}\n"
        f"recall {100 * chosen['recall']:.0f}%, precision {100 * chosen['precision']:.0f}%,\n"
        f"{chosen['flagged_pct_of_customers']:.0f}% of customers flagged",
        xy=(chosen["threshold"], 100 * chosen["recall"]), xytext=(0.5, 88),
        arrowprops={"arrowstyle": "-", "color": DARK}, fontsize=10, color=DARK,
        bbox={"facecolor": "white", "edgecolor": "none"})
    ax.set_xlabel("Decision threshold (flag customers with predicted churn probability at or above)")
    ax.set_ylabel("Percent")
    ax.set_ylim(0, 102)
    ax.set_title("A lower threshold catches more churners but flags more customers")
    ax.legend(loc="lower left", facecolor="white", edgecolor="none", framealpha=1)
    save(fig, "08_threshold_tradeoff", t[[
        "threshold", "precision", "recall", "f1", "customers_flagged",
        "flagged_pct_of_customers", "chosen"]])


def chart_drivers():
    orr = read("logistic_odds_ratios").head(10).iloc[::-1]
    perm = read("permutation_importance_boosting").head(8).iloc[::-1]
    nice = lambda s: s.replace("_", " ").capitalize()  # noqa: E731
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    y = np.arange(len(orr))
    colors = [ACCENT if v > 1 else BLUE for v in orr["odds_ratio"]]
    ax.barh(y, orr["odds_ratio"] - 1, left=1, color=colors, height=0.6)
    ax.errorbar(orr["odds_ratio"], y, xerr=[orr["odds_ratio"] - orr["ci_low"],
                                            orr["ci_high"] - orr["odds_ratio"]],
                fmt="none", ecolor=DARK, elinewidth=1, capsize=3)
    ax.set_xscale("log")
    ax.axvline(1, color=DARK, linewidth=1)
    labels = [f"{nice(f)}: {lvl}\n(vs {ref})" if "SD" not in lvl else f"{nice(f)}: {lvl}"
              for f, lvl, ref in zip(orr["feature"], orr["level"], orr["compared_with"])]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks([0.1, 0.25, 0.5, 1, 2, 4])
    ax.set_xticklabels(["0.1", "0.25", "0.5", "1", "2", "4"])
    ax.set_xlabel("Odds ratio of churn (log scale, bars: 95% bootstrap interval)")
    ax.set_title("Logistic regression: odds ratios\n(red = higher odds, blue = lower)", fontsize=11.5)
    ax = axes[1]
    ax.barh(np.arange(len(perm)), perm["auc_drop_mean"], xerr=perm["auc_drop_std"],
            color=GREY, height=0.6, error_kw={"ecolor": DARK, "elinewidth": 1, "capsize": 3})
    ax.barh(len(perm) - 1, perm["auc_drop_mean"].iloc[-1], color=ACCENT, height=0.6)
    ax.set_yticks(np.arange(len(perm)))
    ax.set_yticklabels([nice(f) for f in perm["feature"]])
    ax.set_xlabel("Drop in test ROC-AUC when the feature is shuffled")
    ax.set_title("Gradient boosting: permutation importance\n(test set, 10 shuffles)", fontsize=11.5)
    fig.suptitle("Contract, tenure and internet service drive the predictions in both models",
                 x=0.04, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    data = pd.concat([orr.assign(panel="odds_ratio"),
                      perm.assign(panel="permutation_importance")], ignore_index=True)
    save(fig, "09_top_drivers", data)


def chart_lift():
    d = read("decile_lift_test")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9))
    ax = axes[0]
    base = 100 * d["churned"].sum() / d["customers"].sum()
    ax.bar(d["decile"], 100 * d["actual_churn_rate"], color=[ACCENT if i < 2 else GREY for i in range(len(d))],
           width=0.7, label="Actual churn rate")
    ax.plot(d["decile"], 100 * d["avg_predicted_churn"], color=DARK, marker="o",
            linewidth=1.5, label="Average predicted probability")
    ax.axhline(base, color=DARK, linestyle="--", linewidth=1)
    ax.text(10.4, base + 1.5, f"All customers {base:.1f}%", ha="right", fontsize=9.5, color=DARK)
    ax.set_xticks(d["decile"])
    ax.set_xlabel("Score decile (1 = highest predicted risk)")
    ax.set_ylabel("Churn rate (%)")
    ax.set_title("Actual churn by risk decile\ntracks the predicted risk", fontsize=11.5)
    ax.legend(loc="upper right", facecolor="white", edgecolor="none", framealpha=1)
    ax = axes[1]
    x = np.r_[0, d["cum_customers_pct"]]
    yv = np.r_[0, d["cum_churners_captured_pct"]]
    ax.plot(x, yv, color=BLUE, marker="o", linewidth=2, label="Model ranking")
    ax.plot([0, 100], [0, 100], color=GREY, linestyle="--", label="Random order")
    for k in (10, 20, 30):
        v = d.loc[d["cum_customers_pct"].round().eq(k), "cum_churners_captured_pct"].iloc[0]
        ax.annotate(f"{v:.0f}%", xy=(k, v), xytext=(k + 3, v - 9), fontsize=10, color=DARK)
    ax.set_xlabel("Customers contacted (% of customers, highest scores first)")
    ax.set_ylabel("Churners reached (%)")
    top30 = d.loc[d["cum_customers_pct"].round().eq(30), "cum_churners_captured_pct"].iloc[0]
    ax.set_title(f"Contacting the top 30% reaches\n{top30:.0f}% of churners", fontsize=11.5)
    ax.legend(loc="lower right", facecolor="white", edgecolor="none", framealpha=1)
    fig.suptitle("Lift on the held-out test set (1,409 customers)", x=0.06, ha="left",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "10_lift_and_calibration", d)


def make_model_charts():
    chart_roc_pr()
    chart_threshold()
    chart_drivers()
    chart_lift()
    return ["07_roc_and_pr_curves", "08_threshold_tradeoff", "09_top_drivers",
            "10_lift_and_calibration"]


if __name__ == "__main__":
    print(make_model_charts())
