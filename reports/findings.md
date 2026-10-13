# Findings: Telco customer churn

Based on the IBM Telco Customer Churn sample dataset (7,043 customers). It is sample data, so these findings describe this dataset, not a real company. Every number comes from code in this repository (`python run_project.py`); sources are named under each finding. All differences are associations, not proven causes.

## Summary

Customers who leave are mostly new, on month-to-month contracts and on fiber optic internet. A simple model can rank customers by risk well enough that contacting the top 30% would reach about 65% of the customers who actually left (test set). The three recommendations below are ideas to test, not results: nothing in this project was tried on real customers.

## Findings

### 1. About one customer in four churned, and they carry 30.5% of monthly charges

1,869 of 7,043 customers churned (26.54%). Their monthly charges add up to $139,130.85 of $456,116.60 (30.50%), because churned customers pay more per month on average ($74.44 vs $61.27).

*Why it matters:* churn is not a small leak, and the customers leaving are above average in monthly bill. *Caveat:* monthly charges of past leavers are a size indicator, not lost profit or lifetime value. Source: `reports/tables/analysis/overall_summary.csv`.

### 2. Contract type and tenure are the strongest signals

Month-to-month customers churn at 42.71% (1,655 of 3,875), against 11.27% for one-year and 2.83% for two-year contracts. Customers in their first 12 months churn at 47.44% (n = 2,186) and customers with 49 to 72 months at 9.51%. The first-year group holds 55.48% of all churners, and month-to-month customers hold 88.55%. In the logistic regression, after adjusting for the other features, a two-year contract has odds of churn 0.26 times those of month-to-month (95% bootstrap interval 0.18 to 0.36); one year: 0.49 (0.40 to 0.60). Contract and tenure are also the strongest features in the statistical tests (Cramér's V 0.41 for contract; rank-biserial -0.48 for tenure) and in the gradient boosting permutation importance (tenure first, contract third).

*Business reading:* customers on short contracts can leave at any time, and the first year seems to be when they decide. *Caveat:* tenure is partly mechanical (someone who left early cannot have a long tenure), and customers who chose a long contract may already be more loyal, so contract may partly reflect customer type rather than cause staying. Sources: `analysis/churn_by_feature.csv`, `analysis/statistical_tests.csv`, `model/logistic_odds_ratios.csv`, `model/permutation_importance_boosting.csv`.

### 3. Fiber optic customers churn about twice as often as DSL customers, even after adjusting

Fiber optic: 41.89% (n = 3,096), DSL 18.96% (n = 2,421), no internet 7.40% (n = 1,526). The gap exists inside every contract type (month-to-month: fiber 54.61% vs DSL 32.22%; two-year: 7.23% vs 1.91%). Month-to-month fiber customers are the highest-risk large segment: 54.61% churn, 2,128 customers (30.2% of all customers), 62.17% of all churners and 40.60% of monthly charges. In the logistic regression the odds of churn for fiber optic are 2.50 times those of DSL (2.08 to 3.11) after adjusting for contract, tenure and the other features, and internet service is second in the permutation importance.

*Business reading:* the fiber product is where the churn is concentrated, and the pattern does not go away when contract and tenure are accounted for. The data cannot say why (price, reliability, competitor offers or a different customer mix). Sources: `analysis/segments_contract_internet.csv`, `analysis/highest_risk_segments.csv`, `model/logistic_odds_ratios.csv`.

### 4. Customers without security and tech-support add-ons, and those paying by electronic check, churn more

Online security: 41.77% churn without it (n = 3,498) vs 14.61% with it (n = 2,019). Tech support: 41.64% without (n = 3,473) vs 15.17% with (n = 2,044). Electronic check: 45.29% (n = 2,365) vs 15.24% to 19.11% for the other three payment methods. After adjusting for the other features the gaps shrink but keep their direction: online security 0.68 (0.58 to 0.79), tech support 0.71 (0.59 to 0.87), electronic check 1.46 (1.20 to 1.78). Gender (V = 0.009) and phone service show no significant association after correction.

*Business reading:* support and security add-ons are linked with staying, and a payment method with manual steps is linked with leaving. *Caveat:* customers who take add-ons may be more engaged to start with, so taking the add-on may not be what keeps them. Sources: `analysis/churn_by_feature.csv`, `analysis/statistical_tests.csv`, `model/logistic_odds_ratios.csv`.

### 5. A simple model can rank customers by churn risk

On 1,409 held-out customers the logistic regression reaches ROC-AUC 0.839 (cross-validation on the training set: 0.845 +/- 0.012) and PR-AUC 0.630, against 0.500 and 0.265 for the baseline that always predicts "stayed". Gradient boosting is similar (test ROC-AUC 0.846, PR-AUC 0.659), so the simpler model was kept. With the threshold of 0.35, chosen on training data to catch about 70% of churners, the test set shows recall 70.6% (264 of 374 churners) and precision 54.4%, with 34.4% of customers flagged. Ranking by score, the top 10% of customers hold 27.5% of the churners (2.75 times the average churn rate), the top 20% hold 49.7% and the top 30% hold 64.7%.

Scoring the 5,174 customers who had not churned (each by a model that did not see them) gives 997 in the High tier (monthly charges $76,354.95), 959 Medium and 3,218 Low. The estimated monthly charges at risk (monthly charges x predicted probability) are $41,288.68 in High, $15,609.60 in Medium and $10,971.70 in Low, $67,869.98 in total, which is 21.4% of these customers' monthly charges. This is an estimate that depends on the predicted probabilities being accurate (calibration is close in nine of ten score deciles, with the third decile over-predicted by about 7 points) and these customers are labelled "stayed" in the snapshot, so the tiers show who looks like past churners, not a forecast. Sources: `model/model_comparison.csv`, `model/threshold_analysis_test.csv`, `model/decile_lift_test.csv`, `model/risk_tiers_current_customers.csv`; method in [docs/model_notes.md](../docs/model_notes.md).

## Recommendations

These follow from the evidence above. None has been tested here, and each needs a measured trial before anyone relies on it.

1. **Offer High-tier month-to-month customers a move to a one- or two-year contract.** Evidence: findings 2 and 5 (month-to-month churn 42.71% vs 2.83% on two-year contracts; the High tier has 997 customers). *How to measure:* randomly split eligible High-tier month-to-month customers into an offer group and a control group that gets no offer. After three to six months compare churn rate and revenue per customer (net of the discount) between the groups. Because customers who accept a contract are self-selected, only the randomised comparison, not "accepters vs others", tells whether the offer works.
2. **Run an early-life check-in programme for customers in their first year.** Evidence: finding 2 (47.44% churn in the first year, 55.48% of churners). Examples: a call after the first bill, help with setup, a review of the plan. *How to measure:* randomly assign new customers to the programme or to normal handling and compare 12-month churn and contact cost.
3. **Investigate fiber optic service and trial a tech-support or security add-on for fiber customers.** Evidence: findings 3 and 4 (fiber 41.89% vs DSL 18.96%; add-on customers churn less). *How to measure:* first collect exit reasons and service-quality data for fiber customers, then randomise a free three-month add-on trial among fiber customers and compare churn with a control group. Do not assume the add-on itself reduces churn: in this data it could simply mark more engaged customers.

## Further analysis needed

- Dates of joining and leaving, to model time to churn (survival analysis) and to test the model on a later period.
- Reasons for leaving (exit survey, support tickets) to find causes behind the associations.
- Costs of an offer, and a way to estimate which customers are persuadable (uplift modelling or an experiment), because a high churn risk does not mean a retention offer will work.
- Customer lifetime value and margin per plan, so priorities use profit and not monthly bills.
- Competitor prices and network quality for the fiber segment.

## Limitations

- **Sample data.** The IBM Telco dataset is a published sample, so results describe this dataset only and not any real company or market.
- **Single snapshot with no dates.** Churn cannot be shown over time, the model has not been tested on a later period, and "current customers" are simply the ones labelled as not churned.
- **No reasons for churn.** Associations cannot be turned into causes.
- **Tenure is partly mechanical.** Customers who left early have short tenure by definition, which inflates the tenure signal.
- **Observational data.** Features overlap (new customers are mostly month-to-month; fiber customers pay more), and adjusted odds ratios only control for the features in the data.
- **Model not validated on new data.** Only a random 20% test set from the same snapshot (374 churners) was used; results may be a few points different elsewhere.
- **Revenue figures are rough.** Monthly charges are not lifetime value or profit, and the at-risk estimate assumes accurate probabilities.
