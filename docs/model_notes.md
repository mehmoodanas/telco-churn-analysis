# Model notes (Day 3: churn prediction)

All numbers come from `python run_project.py` (code in [`src/model.py`](../src/model.py), tables in `reports/tables/model/`, charts 07 to 10 in `reports/figures/`). The data is the IBM Telco sample dataset (7,043 customers), so everything describes this dataset, not a real company. The models show predictive signal and associations. They do not show why customers leave.

## Goal

Rank customers by how likely they are to churn, so that a limited retention effort could be aimed at the highest-risk group. The target is `churned` (1 = left). 1,869 of 7,043 customers (26.54%) churned.

## Method

1. **Split.** 80% training (5,634 customers) and 20% test (1,409), stratified on churn so both have 26.54% churners. Random seed 42 everywhere, so reruns give the same tables.
2. **Preprocessing inside a scikit-learn `Pipeline`.** Numeric columns are standardised, categorical columns one-hot encoded. Because this happens inside the pipeline, the scaler and encoder are fitted on training data only (inside each cross-validation fold too), so nothing from the test set or from the validation fold leaks in.
3. **Models.** (a) a baseline that always predicts "stayed", (b) logistic regression, (c) gradient boosting (`HistGradientBoostingClassifier`).
4. **Tuning.** A small grid per model (logistic regression: `C` in 0.01, 0.1, 1, 10; gradient boosting: depth 2 or 3, learning rate 0.05 or 0.1, 100 or 200 iterations), scored by ROC-AUC with stratified 5-fold cross-validation on the training set only. The grid results are in `cv_grid_results.csv`. Logistic regression picked `C = 10`, the edge of its grid, but the CV ROC-AUC across the whole grid only moves between 0.842 and 0.845, so a wider grid would not change the picture. The reported CV mean and standard deviation are for the chosen settings, so they are a little optimistic because the same folds picked the settings.
5. **Test set.** Used once, for the final numbers. The threshold and the model choice were made without it.

## Feature decisions

| Decision | Reason |
|---|---|
| `customer_id`, `churn`, `churned` not used | Identifier and the target itself. |
| `total_charges` left out | It is about tenure x monthly charges, so it is mostly a copy of two other features (and a result of how long the customer has stayed). Adding it raised the logistic regression test ROC-AUC from 0.839 to 0.843 (`feature_sensitivity.csv`), a small gain that does not justify a column that repeats others. |
| `monthly_charges` left out | The service columns explain 99.9% of its variance (R-squared 0.999, training set). With it in, the odds ratios became unstable (for example fiber optic 9.8 with an interval of 1.3 to 29.5, and a monthly-charges ratio of 0.19, in an earlier run). Dropping it left cross-validated ROC-AUC unchanged (0.845 either way). It is still used for the revenue-at-risk table. |
| `tenure_group`, `is_new_customer`, `addon_services` left out | Derived from `tenure_months` and from the six add-on columns. |
| "No internet service" and "No phone service" merged into "No" in the seven add-on and `multiple_lines` columns | `internet_service` ("No") and `phone_service` ("No") already say the customer has no such service. Keeping a third value would repeat that information and make the odds ratios impossible to separate. After merging, "Yes" on an add-on means "has it, compared with a customer of the same internet type who does not". |
| `tenure_months` kept | The strongest single signal in Day 2. It is partly mechanical: a customer who left early cannot have a long tenure, so the model partly learns "has this customer been with us long", which is not something a company can change. |

Reference categories for the odds ratios: Female, "No" for every Yes/No column, DSL for internet service, Month-to-month for contract, Bank transfer (automatic) for payment method. Tenure is compared per +1 standard deviation (24.6 months).

## Results (held-out test set, 1,409 customers)

| Model | CV ROC-AUC (mean +/- SD) | Test ROC-AUC | Test PR-AUC | Accuracy at 0.5 | Precision at 0.5 | Recall at 0.5 |
|---|---|---|---|---|---|---|
| Baseline (always "stayed") | 0.500 +/- 0.000 | 0.500 | 0.265 | 73.5% | 0 | 0 |
| Logistic regression | 0.845 +/- 0.012 | 0.839 | 0.630 | 79.7% | 63.6% | 55.1% |
| Gradient boosting | 0.850 +/- 0.012 | 0.846 | 0.659 | 80.4% | 67.0% | 51.6% |

**Why accuracy misleads.** 73.5% of customers stay, so a model that never flags anyone is 73.5% accurate and catches no churners. The logistic regression is 79.7% accurate, only 6 points above doing nothing. ROC-AUC (how often a random churner is ranked above a random stayer) and PR-AUC (average precision, whose no-skill value equals the churn rate, 0.265) show the real gain, and precision and recall show what happens to the churners.

**Which model.** Gradient boosting is ahead by 0.005 in CV ROC-AUC and 0.006 on the test set, less than half of one CV standard deviation (0.012). The rule fixed in `choose_final_model` is: use logistic regression unless boosting is better by more than 0.01 CV ROC-AUC. It is not, so logistic regression is the final model, because it is simpler, shows the direction of each association, and is well calibrated. Boosting has the better PR-AUC (0.659 vs 0.630), which a team could weigh differently.

## Threshold

A threshold turns the probability into a flag. The rule: **the highest threshold whose out-of-fold recall on the training set is at least 70%.** The business reason: the assumed action is a low-cost retention offer (a call or discount), while a lost customer loses a monthly bill, so catching most churners matters more than avoiding every unnecessary offer. 70% is a judgement, not a calculated optimum, because the dataset has no information about the cost of an offer or the chance that it works. The rule gives **0.35**.

On the test set, threshold 0.35 gives (confusion matrix: 814 stayers not flagged, 221 stayers flagged, 110 churners missed, 264 churners caught):

- recall 70.6% (264 of 374 churners), precision 54.4%, F1 0.615
- 485 customers flagged, 34.4% of all customers

At the default 0.5, the model catches 55.1% of churners with 63.6% precision. The full table is in `threshold_analysis_test.csv` (chart 08). A team with a costlier offer should move the threshold up; one with a very cheap offer, down.

## Calibration and lift

Test customers were sorted by predicted probability and cut into ten groups of about 141 (`decile_lift_test.csv`, chart 10). Average predicted and actual churn are within 3 percentage points in nine of the ten groups. The exception is the third decile, where the model predicts 47.0% and 39.7% actually churned (7.3 points lower). Top decile: 73.0% actual churn, 2.75 times the average, containing 27.5% of all churners. The top 20% hold 49.7% of churners and the top 30% hold 64.7%. The bottom three deciles hold 2.7% of churners (10 of 374).

## Drivers

- **Logistic regression odds ratios** (`logistic_odds_ratios.csv`, with 95% intervals from 200 bootstrap refits). Two-year contract 0.26 (0.18 to 0.36) and one-year 0.49 (0.40 to 0.60) vs month-to-month; fiber optic 2.50 (2.08 to 3.11) and no internet 0.44 (0.32 to 0.60) vs DSL; +1 SD of tenure 0.45 (0.39 to 0.50); electronic check 1.46 (1.20 to 1.78) vs bank transfer; paperless billing 1.45 (1.27 to 1.71); online security 0.68 (0.58 to 0.79), tech support 0.71 (0.59 to 0.87). Gender, partner and device protection are close to 1.0.
- **Permutation importance, gradient boosting, test set** (`permutation_importance_boosting.csv`). Shuffling `tenure_months` lowers the test ROC-AUC by 0.085, `internet_service` by 0.052 and `contract` by 0.048. Every other feature lowers it by less than 0.004.
- **Compared with Day 2.** The same three groups lead (contract, tenure, internet service), and gender and partner stay near zero. Payment method, online security and tech support ranked high one at a time in Day 2, but in the models their permutation importance is small (below 0.004), because they overlap with contract, tenure and internet type. Their odds ratios still point the same way (electronic check 1.46, online security 0.68, tech support 0.71). Paperless billing had 33.57% churn vs 16.33% without it in Day 2, and its odds ratio after adjusting for the other features is a weaker 1.45. These odds ratios are associations adjusted for the other features in the model, not the effect of changing a customer's contract or payment method.

## Business view

Each of the 7,043 customers was scored by a model that was not fitted on them (5-fold out-of-fold scores), and the 5,174 customers with `churned = 0` were put in tiers (`risk_tiers_current_customers.csv`):

| Tier | Rule (predicted probability) | Customers | Monthly charges | Expected monthly charges at risk |
|---|---|---|---|---|
| High | 0.35 or more | 997 | $76,354.95 | $41,288.68 |
| Medium | 0.175 to below 0.35 | 959 | $61,003.05 | $15,609.60 |
| Low | below 0.175 | 3,218 | $179,627.75 | $10,971.70 |

"Expected monthly charges at risk" is the sum of monthly charges x predicted probability. In total it is $67,869.98 of $316,985.75 monthly charges of these customers (21.4%), and 60.8% of it sits in the High tier, which has 19.3% of the customers.

Assumptions and cautions:

- It is an estimate that assumes the predicted probabilities are accurate. The calibration check above supports that only roughly.
- The 5,174 customers are labelled "stayed" in this snapshot. In real use, a model would score active customers whose outcome is unknown. Here the tiers show which customers look like those who left, not who will leave.
- It is a monthly bill, not lifetime value or profit, and there are no dates, so it does not say when anyone would leave.
- Customer-level scores are not saved in the repository.

## What the model should not be used for

- **Not for causes.** A fiber optic customer has higher churn odds, but the data cannot say whether the product, price or the competition in those areas is the reason. Changing a customer's contract would not necessarily change their risk.
- **Not for deciding who to treat differently on personal traits.** Gender and senior citizen are among the model inputs. The model should not be used to deny service or set prices for individuals.
- **Not for a real company without validation.** It is trained on one sample snapshot with no dates. It has not been tested on a later period, so drift, new products or changed pricing could break it.
- **Not as proof that a retention offer works.** The model ranks customers. Whether an offer keeps them needs a controlled test.
- **Not for the exact revenue forecast.** See the cautions above.

## Limitations

- Single snapshot, no dates, no reason for leaving, no customer service or usage data.
- Cross-validation and the test set both come from the same snapshot, so they do not measure performance on a future period.
- Tenure is partly mechanical (see above).
- 1,409 test customers (374 churners) means the test metrics have uncertainty of a few points. The CV standard deviation of ROC-AUC (0.012) gives a rough idea; no confidence intervals were computed for the test metrics.
- Small, light tuning grids: other settings or models could do somewhat better.
