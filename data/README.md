# Data

## Source

**Telco Customer Churn** (IBM sample dataset)

- Public copy used by this project: https://github.com/IBM/telco-customer-churn-on-icp4d (file `data/Telco-Customer-Churn.csv`, repository licence Apache-2.0)
- The same dataset is widely used from Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- 7,043 rows, 21 columns, about 970 KB
- It is sample data published by IBM for demonstrations. It is not presented here as the records of a real company, and it has no dates, so it is a single snapshot of customers.

The raw file is **not** stored in this repository.

## Download

Automatic (no account needed):

```
python src/download_data.py
```

This saves `data/raw/Telco-Customer-Churn.csv` and checks its SHA-256 hash, so the analysis always runs on the same file. `python run_project.py` does this step for you.

Manual: download the CSV from either link above and save it as `data/raw/Telco-Customer-Churn.csv`.

## Files

| File | Description |
|---|---|
| `data/raw/Telco-Customer-Churn.csv` | Raw file, never modified |
| `data/processed/churn_clean.csv` | Cleaned file created by `src/clean_data.py` (not stored in the repository) |

Column descriptions are in [docs/data_dictionary.md](../docs/data_dictionary.md).
