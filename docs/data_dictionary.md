# Data Dictionary

Describes the cleaned file `data/processed/churn_clean.csv`, created by [`src/clean_data.py`](../src/clean_data.py).

- **Grain:** one row per customer (7,043 rows)
- **Primary key:** `customer_id` (unique)
- **Target:** `churned` (1 = the customer left, 0 = stayed)
- The file is a single snapshot. It has no dates, so it shows who has left, not when.

All counts below come from the clean file.

## Customer

| Column | Raw name | Type | Description | Values |
|---|---|---|---|---|
| `customer_id` | customerID | text | Customer identifier | 7,043 unique values |
| `gender` | gender | text | Customer gender | Female 3,488 · Male 3,555 |
| `senior_citizen` | SeniorCitizen | text | Is the customer a senior citizen (raw file uses 0/1) | No 5,901 · Yes 1,142 |
| `partner` | Partner | text | Has a partner | No 3,641 · Yes 3,402 |
| `dependents` | Dependents | text | Has dependents | No 4,933 · Yes 2,110 |

## Account

| Column | Raw name | Type | Description | Values |
|---|---|---|---|---|
| `tenure_months` | tenure | integer | Months the customer has stayed with the company | 0 to 72, median 29 |
| `contract` | Contract | text | Contract type | Month-to-month 3,875 · One year 1,473 · Two year 1,695 |
| `paperless_billing` | PaperlessBilling | text | Uses paperless billing | No 2,872 · Yes 4,171 |
| `payment_method` | PaymentMethod | text | How the customer pays | Electronic check 2,365 · Mailed check 1,612 · Bank transfer (automatic) 1,544 · Credit card (automatic) 1,522 |
| `monthly_charges` | MonthlyCharges | decimal | Amount charged per month | 18.25 to 118.75, median 70.35 |
| `total_charges` | TotalCharges | decimal | Total amount charged so far | 0 to 8,684.80, median 1,394.55. Set to 0 for 11 new customers (see data quality report) |

The currency is not stated in the source file.

## Services

| Column | Raw name | Description | Values |
|---|---|---|---|
| `phone_service` | PhoneService | Has phone service | No 682 · Yes 6,361 |
| `multiple_lines` | MultipleLines | Has more than one phone line | No 3,390 · Yes 2,971 · No phone service 682 |
| `internet_service` | InternetService | Type of internet service | DSL 2,421 · Fiber optic 3,096 · No 1,526 |
| `online_security` | OnlineSecurity | Internet add-on | No 3,498 · Yes 2,019 · No internet service 1,526 |
| `online_backup` | OnlineBackup | Internet add-on | No 3,088 · Yes 2,429 · No internet service 1,526 |
| `device_protection` | DeviceProtection | Internet add-on | No 3,095 · Yes 2,422 · No internet service 1,526 |
| `tech_support` | TechSupport | Internet add-on | No 3,473 · Yes 2,044 · No internet service 1,526 |
| `streaming_tv` | StreamingTV | Internet add-on | No 2,810 · Yes 2,707 · No internet service 1,526 |
| `streaming_movies` | StreamingMovies | Internet add-on | No 2,785 · Yes 2,732 · No internet service 1,526 |

Every customer has phone service, internet service or both. The six add-ons are only available to customers with internet service, so "No internet service" is kept as its own value. It is not the same as choosing "No".

## Target

| Column | Raw name | Type | Description | Values |
|---|---|---|---|---|
| `churn` | Churn | text | Did the customer leave | No 5,174 (73.46%) · Yes 1,869 (26.54%) |

## Columns added during cleaning

| Column | Type | Description | Values |
|---|---|---|---|
| `churned` | integer | 1 if `churn` is Yes, else 0. Its average is the churn rate. | 0: 5,174 · 1: 1,869 |
| `is_new_customer` | integer | 1 if `tenure_months` is 0 | 1: 11 customers |
| `tenure_group` | text | Tenure band | 0-12 months 2,186 · 13-24 months 1,024 · 25-48 months 1,594 · 49-72 months 2,239 |
| `addon_services` | integer | Number of the six internet add-ons the customer has | 0: 2,219 · 1: 966 · 2: 1,033 · 3: 1,118 · 4: 852 · 5: 571 · 6: 284 |

Of the 2,219 customers with 0 add-ons, 1,526 have no internet service and 693 have internet but chose no add-ons.
