"""Churn prediction models. Writes tables to reports/tables/model/.

Steps (details and reasons in docs/model_notes.md):
    1. Choose features and split the customers 80/20 (stratified, fixed seed).
    2. Dummy baseline, logistic regression and gradient boosting, tuned with a
       small grid and stratified 5-fold cross-validation on the training set only.
    3. Evaluate on the held-out test set.
    4. Choose a decision threshold from training-set (out-of-fold) predictions.
    5. Drivers: logistic regression odds ratios and permutation importance.
    6. Business view: out-of-fold risk scores for current customers, risk tiers
       and a decile lift table.

Every random step uses RANDOM_STATE, so the tables are the same on every run.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_val_predict,
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from clean_data import CheckFailed
from config import TABLES_DIR
from explore import load_clean

OUT_DIR = TABLES_DIR / "model"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
TARGET_RECALL = 0.70      # threshold rule: see choose_threshold()
LR_PREFERENCE_MARGIN = 0.01   # prefer logistic regression unless boosting is clearly better
ROUND = 6

# Features that are left out, and why
EXCLUDED = {
    "customer_id": "identifier",
    "churn": "target (text version)",
    "churned": "target",
    "total_charges": "about tenure x monthly charges, so mostly a copy of two other features",
    "monthly_charges": "99.9% explained by the service columns (R-squared), which makes odds ratios unstable; "
                       "dropping it left the cross-validated AUC unchanged (see feature_sensitivity.csv)",
    "tenure_group": "tenure_months cut into bands",
    "is_new_customer": "tenure_months = 0",
    "addon_services": "count of the six add-on columns",
}

NUMERIC = ["tenure_months"]
# Reference category of each categorical feature (the value the odds ratios compare to)
REFERENCE = {
    "gender": "Female", "senior_citizen": "No", "partner": "No", "dependents": "No",
    "phone_service": "No", "multiple_lines": "No", "internet_service": "DSL",
    "online_security": "No", "online_backup": "No", "device_protection": "No",
    "tech_support": "No", "streaming_tv": "No", "streaming_movies": "No",
    "contract": "Month-to-month", "paperless_billing": "No",
    "payment_method": "Bank transfer (automatic)",
}
CATEGORICAL = list(REFERENCE)
FEATURES = NUMERIC + CATEGORICAL

# "No internet service" / "No phone service" repeat what internet_service and
# phone_service already say, so for these columns they are merged into "No".
MERGE_INTO_NO = ["multiple_lines", "online_security", "online_backup",
                 "device_protection", "tech_support", "streaming_tv", "streaming_movies"]


def prepare_features(df):
    """Feature table with the 'No ... service' values merged into 'No'."""
    X = df[FEATURES].copy()
    for col in MERGE_INTO_NO:
        X[col] = X[col].replace({"No internet service": "No", "No phone service": "No"})
    return X


def make_preprocessor():
    encoder = OneHotEncoder(drop=[REFERENCE[c] for c in CATEGORICAL],
                            handle_unknown="error", sparse_output=False)
    return ColumnTransformer([("num", StandardScaler(), NUMERIC),
                              ("cat", encoder, CATEGORICAL)])


def make_pipeline(model):
    return Pipeline([("prep", make_preprocessor()), ("model", model)])


def candidates():
    """Model name -> (pipeline, parameter grid). Small grids on purpose."""
    return {
        "Baseline (always predicts stayed)": (
            make_pipeline(DummyClassifier(strategy="most_frequent")), {}),
        "Logistic regression": (
            make_pipeline(LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
            {"model__C": [0.01, 0.1, 1, 10]}),
        "Gradient boosting": (
            make_pipeline(HistGradientBoostingClassifier(
                early_stopping=False, random_state=RANDOM_STATE)),
            {"model__max_depth": [2, 3], "model__learning_rate": [0.05, 0.1],
             "model__max_iter": [100, 200]}),
    }


BASELINE, LOGREG, BOOST = list(candidates())


def split_data(df):
    X, y = prepare_features(df), df["churned"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    return X_tr, X_te, y_tr, y_te


def fit_models(X_tr, y_tr):
    """Tune each model by cross-validation on the training set; return fitted models."""
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fitted, grid_rows, cv_rows = {}, [], []
    for name, (pipe, grid) in candidates().items():
        if grid:
            search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, refit=True)
            search.fit(X_tr, y_tr)
            best, params = search.best_estimator_, search.best_params_
            res = pd.DataFrame(search.cv_results_)
            for _, r in res.iterrows():
                grid_rows.append({"model": name, "parameters": str(r["params"]),
                                  "cv_roc_auc_mean": r["mean_test_score"],
                                  "cv_roc_auc_std": r["std_test_score"],
                                  "chosen": int(r["rank_test_score"] == 1)})
        else:
            best, params = pipe.fit(X_tr, y_tr), {}
        fitted[name] = best
        cvr = cross_validate(best, X_tr, y_tr, cv=cv, scoring={
            "roc_auc": "roc_auc", "pr_auc": "average_precision"})
        cv_rows.append({
            "model": name, "chosen_parameters": str(params),
            "cv_roc_auc_mean": cvr["test_roc_auc"].mean(),
            "cv_roc_auc_std": cvr["test_roc_auc"].std(),
            "cv_pr_auc_mean": cvr["test_pr_auc"].mean(),
            "cv_pr_auc_std": cvr["test_pr_auc"].std()})
    return fitted, pd.DataFrame(cv_rows), pd.DataFrame(grid_rows), cv


def metrics_at(y_true, score, threshold):
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {"threshold": threshold, "accuracy": (tp + tn) / len(y_true),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
            "true_positive": tp, "false_positive": fp,
            "false_negative": fn, "true_negative": tn,
            "customers_flagged": tp + fp}


def test_scores(fitted, X_te):
    return {name: m.predict_proba(X_te)[:, 1] for name, m in fitted.items()}


def evaluate(scores, y_te, cv_table):
    """Test-set metrics at threshold 0.5 plus threshold-free AUCs, with CV results."""
    rows = []
    for name, s in scores.items():
        row = {"model": name,
               "test_roc_auc": roc_auc_score(y_te, s),
               "test_pr_auc": average_precision_score(y_te, s)}
        row.update({k: v for k, v in metrics_at(y_te, s, 0.5).items() if k != "threshold"})
        rows.append(row)
    out = pd.DataFrame(rows)
    return cv_table[["model", "cv_roc_auc_mean", "cv_roc_auc_std",
                     "cv_pr_auc_mean", "cv_pr_auc_std"]].merge(out, on="model")


def choose_final_model(cv_table):
    """Logistic regression unless boosting beats it by more than the margin (CV ROC-AUC)."""
    auc = cv_table.set_index("model")["cv_roc_auc_mean"]
    if auc[BOOST] - auc[LOGREG] > LR_PREFERENCE_MARGIN:
        return BOOST
    return LOGREG


def choose_threshold(model, X_tr, y_tr, cv):
    """Highest threshold whose out-of-fold recall on the training set is at least
    TARGET_RECALL. Chosen without looking at the test set. Returns (threshold, table)."""
    oof = cross_val_predict(model, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    grid = np.round(np.arange(0.05, 0.85, 0.05), 2)
    table = pd.DataFrame([metrics_at(y_tr, oof, t) for t in grid])
    ok = table[table["recall"] >= TARGET_RECALL]
    return float(ok["threshold"].max()), table


def threshold_table(score, y_te, chosen):
    grid = np.round(np.arange(0.05, 0.85, 0.05), 2)
    table = pd.DataFrame([metrics_at(y_te, score, t) for t in grid])
    table["flagged_pct_of_customers"] = 100 * table["customers_flagged"] / len(y_te)
    table["chosen"] = (table["threshold"] == chosen).astype(int)
    return table


def decile_table(y, score, n_groups=10):
    """Customers sorted by score and cut into equal groups: actual vs predicted churn."""
    d = pd.DataFrame({"y": np.asarray(y), "score": score}).sort_values(
        "score", ascending=False, kind="mergesort").reset_index(drop=True)
    d["decile"] = (np.arange(len(d)) * n_groups // len(d)) + 1
    g = d.groupby("decile").agg(customers=("y", "size"), churned=("y", "sum"),
                                avg_predicted_churn=("score", "mean")).reset_index()
    base = d["y"].mean()
    g["actual_churn_rate"] = g["churned"] / g["customers"]
    g["lift"] = g["actual_churn_rate"] / base
    g["cum_customers_pct"] = 100 * g["customers"].cumsum() / len(d)
    g["cum_churners_captured_pct"] = 100 * g["churned"].cumsum() / d["y"].sum()
    return g


def odds_ratios(model, X_tr, y_tr, n_boot=200):
    """Odds ratios of the logistic regression fitted on the training set.
    95% intervals: percentile of n_boot bootstrap refits (same settings)."""
    def coefs(m):
        names = m.named_steps["prep"].get_feature_names_out()
        return pd.Series(m.named_steps["model"].coef_[0], index=names)

    base = coefs(model)
    rng = np.random.RandomState(RANDOM_STATE)
    boots = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(X_tr), len(X_tr))
        m = make_pipeline(LogisticRegression(
            C=model.named_steps["model"].C, max_iter=2000, random_state=RANDOM_STATE))
        m.fit(X_tr.iloc[idx], y_tr.iloc[idx])
        boots.append(coefs(m))
    boots = pd.DataFrame(boots)
    sd = X_tr[NUMERIC].std()
    rows = []
    for name in base.index:
        kind, rest = name.split("__", 1)
        if kind == "num":
            feature, level = rest, f"per +1 SD ({sd[rest]:.1f})"
            reference = "average customer"
        else:
            feature = next(c for c in CATEGORICAL if rest.startswith(c + "_"))
            level, reference = rest[len(feature) + 1:], REFERENCE[feature]
        rows.append({"feature": feature, "level": level, "compared_with": reference,
                     "odds_ratio": np.exp(base[name]),
                     "ci_low": np.exp(boots[name].quantile(0.025)),
                     "ci_high": np.exp(boots[name].quantile(0.975))})
    out = pd.DataFrame(rows)
    out["distance_from_1"] = np.abs(np.log(out["odds_ratio"]))
    return out.sort_values("distance_from_1", ascending=False).drop(
        columns="distance_from_1").reset_index(drop=True)


def permutation_table(model, X_te, y_te, n_repeats=10):
    r = permutation_importance(model, X_te, y_te, scoring="roc_auc", n_repeats=n_repeats,
                               random_state=RANDOM_STATE)
    out = pd.DataFrame({"feature": X_te.columns, "auc_drop_mean": r.importances_mean,
                        "auc_drop_std": r.importances_std})
    return out.sort_values("auc_drop_mean", ascending=False).reset_index(drop=True)


def risk_tiers(score, df, threshold):
    """Tier customers by predicted churn probability.
    High: at or above the chosen threshold. Medium: from half the threshold up to it.
    Low: below half the threshold."""
    tier = np.where(score >= threshold, "High",
                    np.where(score >= threshold / 2, "Medium", "Low"))
    d = df.assign(score=score, tier=tier, expected=df["monthly_charges"] * score)
    g = d.groupby("tier").agg(
        customers=("score", "size"), avg_predicted_churn=("score", "mean"),
        monthly_charges=("monthly_charges", "sum"),
        expected_monthly_charges_at_risk=("expected", "sum")).reindex(
        ["High", "Medium", "Low"]).reset_index()
    g["share_of_customers_pct"] = 100 * g["customers"] / g["customers"].sum()
    g["share_of_expected_at_risk_pct"] = (
        100 * g["expected_monthly_charges_at_risk"] / g["expected_monthly_charges_at_risk"].sum())
    return g


def out_of_fold_scores(model, X, y, cv):
    """Score every customer with models that did not see that customer."""
    return cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]


def feature_sensitivity(df, X_tr, X_te, y_tr, y_te):
    """Logistic regression test ROC-AUC with and without the two excluded numeric columns.
    Also the R-squared of monthly_charges explained by the other features."""
    rows = []
    for label, extra in (("tenure + categorical (used)", []),
                         ("+ total_charges", ["total_charges"]),
                         ("+ monthly_charges", ["monthly_charges"]),
                         ("+ both", ["total_charges", "monthly_charges"])):
        A, B = X_tr.copy(), X_te.copy()
        for col in extra:
            A[col], B[col] = df.loc[A.index, col], df.loc[B.index, col]
        prep = ColumnTransformer([
            ("num", StandardScaler(), NUMERIC + extra),
            ("cat", OneHotEncoder(drop=[REFERENCE[c] for c in CATEGORICAL],
                                  sparse_output=False), CATEGORICAL)])
        m = Pipeline([("prep", prep), ("model", LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE))]).fit(A, y_tr)
        rows.append({"logistic_regression_features": label,
                     "test_roc_auc": roc_auc_score(y_te, m.predict_proba(B)[:, 1])})
    others = pd.get_dummies(X_tr, drop_first=True).astype(float)
    r2 = LinearRegression().fit(others, df.loc[X_tr.index, "monthly_charges"]).score(
        others, df.loc[X_tr.index, "monthly_charges"])
    rows.append({"logistic_regression_features": "R-squared of monthly_charges on the used features (training set)",
                 "test_roc_auc": r2})
    return pd.DataFrame(rows).rename(columns={"test_roc_auc": "value"})


def check(condition, message, failures):
    if not condition:
        failures.append(message)


def run_model(verbose=True):
    """Fit, evaluate, save tables and run the model checks. Returns a results dict."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_clean()
    X_tr, X_te, y_tr, y_te = split_data(df)
    ids_tr, ids_te = df.loc[X_tr.index, "customer_id"], df.loc[X_te.index, "customer_id"]

    fitted, cv_table, grid_table, cv = fit_models(X_tr, y_tr)
    scores = test_scores(fitted, X_te)
    comparison = evaluate(scores, y_te, cv_table)
    final_name = choose_final_model(cv_table)
    final = fitted[final_name]
    threshold, train_thresholds = choose_threshold(final, X_tr, y_tr, cv)
    thresholds = threshold_table(scores[final_name], y_te, threshold)

    # curves for the charts
    roc_rows, pr_rows = [], []
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve(y_te, s)
        roc_rows.append(pd.DataFrame({"model": name, "false_positive_rate": fpr,
                                      "true_positive_rate": tpr}))
        p, r, _ = precision_recall_curve(y_te, s)
        pr_rows.append(pd.DataFrame({"model": name, "recall": r, "precision": p}))
    roc_df, pr_df = pd.concat(roc_rows), pd.concat(pr_rows)

    # calibration and lift on the test set
    test_deciles = decile_table(y_te, scores[final_name])
    drivers_lr = odds_ratios(fitted[LOGREG], X_tr, y_tr)
    perm_boost = permutation_table(fitted[BOOST], X_te, y_te)
    perm_final = permutation_table(final, X_te, y_te)
    sensitivity = feature_sensitivity(df, X_tr, X_te, y_tr, y_te)

    # business view: out-of-fold scores for every customer, current customers only
    X_all, y_all = prepare_features(df), df["churned"]
    oof = out_of_fold_scores(final, X_all, y_all, cv)
    current = df["churned"] == 0
    tiers = risk_tiers(oof[current.to_numpy()], df[current], threshold)
    cm = confusion_matrix(y_te, (scores[final_name] >= threshold).astype(int))

    assumptions = pd.DataFrame({"assumption": [
        f"Scores for all {len(df):,} customers are out-of-fold: each customer is scored by a model fitted on other customers.",
        "Only customers with churned = 0 are tiered. In this snapshot they are the customers who had not left.",
        f"High tier: predicted churn probability at or above {threshold:.2f} (the chosen threshold). Medium: {threshold / 2:.3f} to below {threshold:.2f}. Low: below {threshold / 2:.3f}.",
        "Expected monthly charges at risk = sum of monthly_charges x predicted churn probability. It assumes the probabilities are well calibrated and is an estimate, not a forecast.",
        "It is a monthly bill, not lifetime value or profit, and the data has no dates, so it says nothing about when customers would leave.",
        "Sample dataset: the figures describe this dataset, not a real company."]})

    tables = {
        "model_comparison": comparison, "cv_grid_results": grid_table,
        "threshold_analysis_test": thresholds, "threshold_analysis_train_oof": train_thresholds,
        "decile_lift_test": test_deciles, "logistic_odds_ratios": drivers_lr,
        "permutation_importance_boosting": perm_boost,
        "permutation_importance_final": perm_final,
        "feature_sensitivity": sensitivity, "risk_tiers_current_customers": tiers,
        "business_assumptions": assumptions,
        "roc_curve": roc_df, "pr_curve": pr_df,
        "feature_decisions": pd.DataFrame(
            [{"feature": c, "decision": "excluded", "reason": r} for c, r in EXCLUDED.items()]
            + [{"feature": c, "decision": "used (numeric, scaled)", "reason": ""} for c in NUMERIC]
            + [{"feature": c, "decision": "used (one-hot, reference: " + REFERENCE[c] + ")",
                "reason": "'No ... service' merged into 'No'" if c in MERGE_INTO_NO else ""}
               for c in CATEGORICAL]),
        "final_model": pd.DataFrame([
            {"measure": "final_model", "value": final_name},
            {"measure": "threshold", "value": threshold},
            {"measure": "threshold_rule", "value": f"highest threshold with out-of-fold recall >= {TARGET_RECALL} on the training set"},
            {"measure": "train_customers", "value": len(X_tr)},
            {"measure": "test_customers", "value": len(X_te)},
            {"measure": "random_state", "value": RANDOM_STATE},
            {"measure": "confusion_tn_fp_fn_tp_at_threshold", "value": " ".join(map(str, cm.ravel()))}]),
    }
    for name, t in tables.items():
        t.round(ROUND).to_csv(OUT_DIR / f"{name}.csv", index=False)

    failures = []
    run_checks(df, ids_tr, ids_te, X_tr, y_tr, y_te, scores, comparison, final_name,
               threshold, thresholds, tiers, current, test_deciles, oof, failures)
    if failures:
        raise CheckFailed("Model checks failed:\n- " + "\n- ".join(failures))

    if verbose:
        print(comparison.round(3).to_string(index=False))
        print(f"final model: {final_name}, threshold {threshold}")
        print(tiers.round(2).to_string(index=False))
    return {"comparison": comparison, "final": final_name, "threshold": threshold,
            "tiers": tiers, "tables": tables}


def run_checks(df, ids_tr, ids_te, X_tr, y_tr, y_te, scores, comparison, final_name,
               threshold, thresholds, tiers, current, test_deciles, oof, failures):
    c = lambda cond, msg: check(cond, msg, failures)  # noqa: E731
    c(set(ids_tr).isdisjoint(set(ids_te)), "train and test customers overlap")
    c(len(ids_tr) + len(ids_te) == len(df), "train + test do not cover all customers")
    c(ids_tr.is_unique and ids_te.is_unique, "duplicate customer ids in train or test")
    c(abs(y_tr.mean() - y_te.mean()) < 0.005, "churn rate differs between train and test")
    c(not set(FEATURES) & {"customer_id", "churn", "churned"}, "target or id in the features")
    c(list(X_tr.columns) == FEATURES, "feature list differs from FEATURES")
    c(not X_tr.isna().any().any(), "missing values in the features")
    comp = comparison.set_index("model")
    c(abs(comp.loc[BASELINE, "test_roc_auc"] - 0.5) < 1e-9, "baseline ROC-AUC is not 0.5")
    for m in (LOGREG, BOOST):
        c(comp.loc[m, "test_roc_auc"] > comp.loc[BASELINE, "test_roc_auc"] + 0.2,
          f"{m}: test ROC-AUC not clearly above the baseline")
        c(comp.loc[m, "test_pr_auc"] > y_te.mean() + 0.2,
          f"{m}: test PR-AUC not clearly above the churn rate")
    for s in scores.values():
        c(np.all((s >= 0) & (s <= 1)), "scores outside 0..1")
    c(np.all((oof >= 0) & (oof <= 1)), "out-of-fold scores outside 0..1")
    c(thresholds["chosen"].sum() == 1, "chosen threshold is not in the threshold table")
    c(tiers["customers"].sum() == current.sum(), "risk tier counts do not add up to current customers")
    c(test_deciles["customers"].sum() == len(y_te) and test_deciles["churned"].sum() == y_te.sum(),
      "decile table does not add up")
    c(test_deciles.loc[0, "lift"] > 1.5, "top decile lift is not above 1.5")
    c(test_deciles["cum_churners_captured_pct"].iloc[-1] > 99.99, "deciles do not capture all churners")
    c(test_deciles["churned"].iloc[0] > test_deciles["churned"].iloc[-1],
      "top decile has no more churners than the bottom decile")
    c(tiers.loc[0, "avg_predicted_churn"] > tiers.loc[2, "avg_predicted_churn"],
      "high tier is not riskier than low tier")
    c(final_name in comp.index, "final model not found")


if __name__ == "__main__":
    run_model()
