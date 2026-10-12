# Analysis notes (Day 2: exploratory analysis and statistical tests)

All numbers come from `python run_project.py` (tables in `reports/tables/analysis/`, charts in `reports/figures/`). The data is the IBM Telco sample dataset (7,043 customers), so results describe this dataset, not a real company. Everything below is an association; none of it shows what causes customers to leave.

## Metric definitions

| Metric | Definition |
|---|---|
| Churn rate | Customers with `churned = 1` divided by customers in the group, as a percentage. |
| Monthly revenue lost | Sum of `monthly_charges` of churned customers. |
| Share of monthly charges lost | Monthly revenue lost divided by the sum of `monthly_charges` of all customers. |
| Share of churners | Churned customers in a group divided by all 1,869 churned customers. |
| Gap (percentage points) | Highest group churn rate minus lowest group churn rate within one feature. |

**Limits of "monthly revenue lost".** It is the monthly bill of customers who already left, taken from the same snapshot. It is not lifetime value, not profit, and not a forecast: it ignores how long each customer would have stayed, costs, and the revenue from customers who joined. Treat it as a size indicator only.

## Methods

- **Group tables.** Each categorical feature is split into its values, with customer count, churn count and churn rate. For every feature the churn counts add up to 1,869 and customers to 7,043 (checked in code).
- **Two-way segments.** Contract x internet service and contract x tenure group. Every cell shows its customer count. Cells with fewer than 50 customers would keep their count but get no churn rate and a `small_cell` flag; no cell was that small in these two tables (the smallest has 68 customers).
- **Highest-risk segments.** Single feature values plus three pairs (contract x internet service, contract x tenure group, internet service x payment method), keeping only segments with at least 100 customers, ranked by churn rate.
- **Categorical features vs churn:** chi-square test of independence (scipy; Yates continuity correction is applied automatically for 2x2 tables). Effect size is Cramér's V, computed without correction: 0 means no association, 1 a perfect one. For a 2-column table I read V below 0.10 as negligible, 0.10 to 0.30 as small, 0.30 to 0.50 as medium (a common rule of thumb, not a law).
- **Numeric features vs churn:** Mann-Whitney U test (two-sided), because tenure and charges are not normally distributed. Effect size is the rank-biserial correlation, 2U / (n1 x n2) - 1, with U for the churned group. Negative means churned customers have lower values.
- **Multiple testing.** 21 tests were run (18 categorical, 3 numeric). With many tests, some small p-values would appear by chance, so p-values are adjusted with the Holm method, and Bonferroni is also reported. Both are in `statistical_tests.csv` (`p_holm`, `p_bonferroni`).

### Why effect size matters more than p-values here

With 7,043 customers, the tests can detect very small differences. Almost every feature has a p-value far below 0.05, and the Holm correction does not change that: 19 of 21 tests remain significant. A tiny p-value only says that a difference is unlikely to be pure chance in this sample. It does not say the difference is large or useful. Cramér's V and the rank-biserial correlation measure how strong the association is, so they are used to rank the features. Features can be significant and still weak (for example `multiple_lines`, V = 0.04, Holm p = 0.010).

## Main results

**Overall.** 1,869 of 7,043 customers churned (26.54%). Their monthly charges total $139,130.85 of $456,116.60, which is 30.50% of all monthly charges. Churned customers pay more on average ($74.44 vs $61.27 a month).

**Contract.** Month-to-month 42.71% (1,655 of 3,875), one year 11.27% (166 of 1,473), two year 2.83% (48 of 1,695). Month-to-month customers are 88.55% of all churners.

**Tenure.** 0-12 months 47.44% (n = 2,186), 13-24 months 28.71% (n = 1,024), 25-48 months 20.39% (n = 1,594), 49-72 months 9.51% (n = 2,239). The first-year group holds 55.48% of all churners. Median tenure is 10 months for churned and 38 for stayed customers (mean 17.98 vs 37.57).

**Internet service.** Fiber optic 41.89% (n = 3,096), DSL 18.96% (n = 2,421), no internet 7.40% (n = 1,526). Fiber optic customers are 69.40% of churners and 62.11% of monthly charges.

**Payment method.** Electronic check 45.29% (n = 2,365), mailed check 19.11% (n = 1,612), bank transfer 16.71% (n = 1,544), credit card 15.24% (n = 1,522).

**Support add-ons.** Online security: 41.77% churn without it (n = 3,498) vs 14.61% with it (n = 2,019). Tech support: 41.64% without (n = 3,473) vs 15.17% with (n = 2,044). The "no internet service" group (n = 1,526, 7.40%) is separate in these tables. The `addon_services` count is hard to read on its own: its 0 group (21.41%, n = 2,219) mixes customers with no internet and customers with internet but no add-ons, so it does not fall steadily.

**Other customer traits.** Senior citizens 41.68% (n = 1,142) vs 23.61% for others. Paperless billing 33.57% vs 16.33% without. No partner 32.96% vs 19.66% with. No dependents 31.28% vs 15.45% with. Gender: female 26.92%, male 26.16%. Phone service: 26.71% with vs 24.93% without.

**Segments.** Month-to-month fiber optic customers churn at 54.61% (1,162 of 2,128), which is 62.17% of all churners and 40.60% of monthly charges. Month-to-month customers in their first year churn at 51.35% (n = 1,994). Fiber optic customers paying by electronic check churn at 53.23% (n = 1,595). Two-year customers in their first two years had 0 churners, but those cells are small (68 and 90 customers).

**Contract x internet service (churn rate, n).** Month-to-month: DSL 32.22% (1,223), fiber 54.61% (2,128), none 18.89% (524). One year: DSL 9.30% (570), fiber 19.29% (539), none 2.47% (364). Two year: DSL 1.91% (628), fiber 7.23% (429), none 0.78% (638). Fiber optic is higher than DSL within every contract type.

**Statistical tests (ranked by effect size).** Strongest: tenure in months (rank-biserial -0.48), contract (V = 0.41), tenure group (0.35), online security (0.35), tech support (0.34), internet service (0.32), payment method (0.30), total charges (rank-biserial -0.30). Monthly charges: rank-biserial 0.24 (churned customers higher). Gender (V = 0.009, Holm p = 0.68) and phone service (V = 0.012, Holm p = 0.68) are the two features with no significant association after correction.

## Cautions

- Features overlap: new customers are mostly on month-to-month contracts (1,994 of 2,186 first-year customers), and fiber optic customers pay more, so the single-feature gaps partly describe the same customers. Day 3 models will help separate them, but still only show associations.
- `total_charges` grows with tenure, so its association with churn largely repeats the tenure result.
- The sample has a single snapshot, with no dates, so no trend over time can be shown.
