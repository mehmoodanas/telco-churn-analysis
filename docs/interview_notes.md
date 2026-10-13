# Interview notes: Telco churn project

Numbers come from `python run_project.py` (see [reports/findings.md](../reports/findings.md) and [docs/model_notes.md](model_notes.md)). The data is a public sample dataset, so I describe results as "in this dataset".

## Five-minute walkthrough

1. **Problem (30 s).** A telecom retention manager wants to know how many customers leave, which groups leave most, whether the differences are real, whether we can predict who is at risk, and what to try first. Dataset: IBM Telco sample, 7,043 customers, 26.54% churned.
2. **Data (45 s).** I read the file as text first, because 11 blank `TotalCharges` values would have silently become missing numbers. All 11 belong to customers with tenure 0, so I set them to 0. I wrote 30 automatic checks (no duplicates, consistent services, row counts) that stop the pipeline if they fail.
3. **Exploration (60 s).** Churned customers pay more per month ($74.44 vs $61.27) and carry 30.5% of monthly charges. Month-to-month customers churn at 42.71% against 2.83% on two-year contracts. The first year of tenure is 47.44%. Fiber optic is 41.89% against DSL 18.96%. I ran 21 statistical tests and used effect sizes (Cramér's V) and Holm-adjusted p-values, because with 7,043 rows nearly everything is significant.
4. **Model (90 s).** Stratified 80/20 split, preprocessing in a Pipeline, baseline plus logistic regression plus gradient boosting, small tuning grids with 5-fold cross-validation on the training set. Logistic regression: test ROC-AUC 0.839, PR-AUC 0.630 (baseline 0.500 and 0.265). Boosting was only slightly better, so I kept the simpler model. Threshold 0.35, chosen on training data to catch about 70% of churners: on the test set that gives recall 70.6%, precision 54.4%, 34.4% of customers flagged.
5. **Business view and limits (45 s).** Contacting the top 30% by score reaches 64.7% of churners. I tiered the non-churned customers into High, Medium and Low and estimated monthly charges at risk. I stress that this is an estimate on a sample dataset with no dates, and the recommendations (contract offers for High-tier month-to-month customers, a first-year programme, a fiber investigation) need a controlled test before anyone relies on them.

## Likely questions

**1. Why not just use accuracy?**
73.5% of customers stay, so a model that always predicts "stayed" is 73.5% accurate and finds no churners. My logistic regression is 79.7% accurate at threshold 0.5, only about 6 points better. So I report ROC-AUC, PR-AUC (whose no-skill value is the churn rate, 0.265), and precision and recall at a chosen threshold.

**2. How did you avoid data leakage?**
The split happens first. Scaling and one-hot encoding sit inside a scikit-learn Pipeline, so they are fitted only on training data (and only on the training folds inside cross-validation). Tuning used the training set only. The threshold was chosen from out-of-fold training predictions, not the test set. I excluded the id and target, and left out `total_charges` because it is mostly tenure x monthly charges. The test set was used once for the final numbers. A check in the code confirms no customer id is in both train and test.

**3. Why logistic regression?**
Boosting was ahead by 0.005 in cross-validated ROC-AUC (0.850 vs 0.845), less than half of the 0.012 standard deviation across folds. I fixed a rule before looking at the test set: keep logistic regression unless boosting is better by more than 0.01. It also gives odds ratios a manager can read, and its probabilities were reasonably calibrated.

**4. How did you choose the threshold?**
By a business rule: the assumed action is a low-cost retention offer, so missing a churner costs more than a wasted offer. I picked the highest threshold whose out-of-fold recall on the training set is at least 70%, which gave 0.35. I say clearly that 70% is a judgement, since the dataset has no cost for an offer or its success rate. A costlier offer should push the threshold up.

**5. What would you do with more data?**
Dates (to use time-to-churn methods and test on a later period), reasons for leaving, offer costs and past offers (to estimate which customers an offer actually helps, with an experiment or uplift model), customer lifetime value, and usage or complaint data.

**6. Why did you drop `monthly_charges` from the model?**
It is explained 99.9% by the service columns. With it included, the odds ratios turned unstable (fiber optic 9.8 with an interval from 1.3 to 29.5). Removing it left cross-validated ROC-AUC unchanged (0.845), so I dropped it from the features, but still used it to size the revenue at risk.

**7. How do you read the odds ratios?**
As adjusted associations. After accounting for the other features, fiber optic customers have 2.50 times the odds of churn of DSL customers, and two-year contracts 0.26 times those of month-to-month. They do not say that changing a customer's contract would change their risk.

**8. What is the difference between significant and important?**
With 7,043 customers 19 of 21 tests were significant after Holm correction, including weak ones. Effect size separates them: contract has Cramér's V 0.41, multiple lines 0.04.

**9. How do you know the model is not just lucky on this test set?**
Cross-validation on the training set gives 0.845 +/- 0.012 and the test set 0.839, which agree. Also the test set has only 374 churners, so I treat the metrics as accurate to within a few points. I did not test on a later period because the data has no dates.

**10. What is your estimate of the revenue at risk, and how much do you trust it?**
Summing monthly charges x predicted churn probability for the 5,174 customers labelled "stayed" gives $67,869.98 a month, 21.4% of their charges, with 60.8% of it in the 997-customer High tier. I trust it as a rough size, not a forecast: it assumes accurate probabilities (the third score decile was over-predicted by about 7 points), is a monthly bill and not profit, and these customers are labelled as stayers in a single snapshot.

## Key decisions

| Decision | Choice | Reason |
|---|---|---|
| Blank `TotalCharges` | Set to 0 | All 11 have tenure 0 (not yet billed). |
| "No internet service" values | Kept in exploration, merged into "No" for the model | `internet_service` already holds that information. |
| `total_charges` | Not a model feature | Mostly tenure x monthly charges. |
| `monthly_charges` | Not a model feature | 99.9% explained by service columns, unstable odds ratios, no loss in AUC. |
| Metrics | ROC-AUC, PR-AUC, precision, recall | Accuracy is 73.5% for a model that finds nothing. |
| Final model | Logistic regression | Boosting not better by more than 0.01 CV AUC. |
| Threshold | 0.35 | Highest threshold with 70% out-of-fold recall (low-cost offer). |
| Risk scores | Out-of-fold | Every customer is scored by a model that never saw them. |
| Customer-level scores | Not committed | Keeps the repository to code, tables and charts. |

## Three honest CV bullet points

- Built a reproducible Python pipeline (pandas, scikit-learn) that downloads, cleans and checks a 7,043-row customer dataset with 30 automatic validation checks, runs statistical tests with effect sizes and multiple-testing correction, and rebuilds all tables and charts from one command.
- Trained and evaluated churn models (logistic regression, gradient boosting) with stratified cross-validation and a held-out test set, reaching test ROC-AUC 0.84 (baseline 0.50), and chose a decision threshold on training data for a stated business goal (70% recall).
- Wrote a findings report and model notes that turn the results into three testable recommendations with measurement plans, and state the limits (sample data, one snapshot, associations only).

*Learning portfolio project. It was not done for an employer and no recommendation was tested on customers.*
