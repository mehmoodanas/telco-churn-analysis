# Data Quality Report

How the raw Telco churn file was inspected and cleaned, and what its limits are. All counts come from running `python run_project.py`. Supporting tables are in [`tables/inspection/`](tables/inspection/) and [`tables/cleaning/`](tables/cleaning/).

## Summary

| Item | Value |
|---|---|
| Rows in raw file | 7,043 |
| Rows in clean file | 7,043 (no rows removed) |
| Columns | 21 raw, 25 clean (4 added for analysis) |
| Duplicate customers | 0 |
| Missing values | 11, all in `TotalCharges` |
| Customers who churned | 1,869 (26.54%) |
| Cleaning checks | 30, all passed |

The file is in good condition. The only real data problem is 11 blank `TotalCharges` values, and they have a clear explanation.

## Inspection results

The raw file is read with every column as text, so nothing is silently converted before it is checked.

| Check | Rows failing |
|---|---|
| Duplicate `customerID` | 0 |
| Duplicate full rows | 0 |
| No internet, but an add-on is not "No internet service" | 0 |
| Has internet, but an add-on says "No internet service" | 0 |
| No phone, but `MultipleLines` is not "No phone service" | 0 |
| Has phone, but `MultipleLines` says "No phone service" | 0 |
| Blank `TotalCharges` with tenure above 0 | 0 |
| Tenure 0 with a `TotalCharges` value | 0 |
| Customer with neither phone nor internet | 0 |
| Text values with extra spaces (all columns except `TotalCharges`) | 0 |

Every category column contains only its expected values (for example, `Contract` has exactly three values).

## Cleaning rules

### 1. Blank TotalCharges (11 customers)

`TotalCharges` is blank for 11 customers. All 11 have a tenure of 0 months, and no customer with tenure 0 has a value. They are new customers who have not been billed yet (10 on two-year contracts, 1 on a one-year contract, none churned).

**Rule:** set `total_charges` to 0 and mark them with `is_new_customer = 1`.

Why not delete them? They are real customers, and deleting them would slightly change the churn rate. Why not fill with the average? The true amount charged so far is known to be 0.

### 2. Data types

- `tenure`, `MonthlyCharges` and `TotalCharges` are converted to numbers. Apart from the 11 blanks, every value is numeric.
- `SeniorCitizen` is stored as 0/1 in the raw file while every other yes/no column uses Yes/No. It is converted to Yes/No so all columns read the same way.

### 3. "No internet service" and "No phone service"

These values are **kept as they are**. A customer with no internet cannot choose online security, which is different from a customer who has internet and chose not to buy it. Merging them into "No" would hide that difference. The modelling stage will decide how to encode them.

### 4. Column names

Converted to snake_case (`MonthlyCharges` → `monthly_charges`, `tenure` → `tenure_months`). The full mapping is in [`docs/data_dictionary.md`](../docs/data_dictionary.md).

### 5. Columns added

| Column | Rule |
|---|---|
| `churned` | 1 if `churn` is Yes, else 0 |
| `is_new_customer` | 1 if `tenure_months` is 0 |
| `tenure_group` | 0-12, 13-24, 25-48, 49-72 months |
| `addon_services` | Count of "Yes" across the six internet add-ons (0 to 6) |

### 6. Outliers

No values are removed or capped. `monthly_charges` ranges from 18.25 to 118.75 and `tenure_months` from 0 to 72, which are all plausible for a telecom customer.

## Validation

`src/clean_data.py` runs 30 checks and stops without saving if any of them fail. They cover:

- row count equals the raw file and the expected 7,043
- `customer_id` is unique and no column has missing values
- numeric ranges (tenure 0 to 72, monthly charges above 0)
- `total_charges` is 0 only for new customers
- churn count and the sum of monthly charges are unchanged from the raw file
- every text column contains only allowed values

The checks were tested by adding a duplicate row and an invalid contract value on purpose; both were caught. Results are saved in [`tables/cleaning/validation_checks.csv`](tables/cleaning/validation_checks.csv).

## TotalCharges compared with tenure × MonthlyCharges

`TotalCharges` is not exactly `tenure × MonthlyCharges`. For the 7,032 customers with a value, the median difference is 2.0%, the 90th percentile is 7.52% and the largest is 45.06%. 6,646 customers are within 10%.

This is expected: `MonthlyCharges` is the current price, and a customer's price can change over time when they add or drop services. `TotalCharges` is therefore **not corrected**. It also means `total_charges` carries almost the same information as tenure and monthly charges combined, which matters for modelling.

## Limitations

1. **Sample data.** The dataset was published by IBM for demonstrations. Findings describe this dataset and are not evidence about any real company.
2. **One snapshot, no dates.** The file does not say when a customer joined or left, so churn over time, seasonality and survival analysis are not possible. `tenure_months` is the only time information.
3. **No reason for churn.** The file does not record why customers left, so explanations in this project are based on associations with customer attributes.
4. **Customers who left have shorter tenure by definition.** Tenure stops counting when a customer leaves, so tenure and churn are linked in a way that is partly mechanical.
5. **New customers cannot have churned yet.** The 11 customers with tenure 0 are all non-churned.
6. **Class imbalance.** 26.54% of customers churned. A model that predicts "no churn" for everyone would be 73.46% accurate, so accuracy alone is a poor measure.
7. **Currency is not stated** in the source file.
8. **No customer value beyond charges.** There is no cost or margin data, so "revenue at risk" can only be measured with monthly charges.
