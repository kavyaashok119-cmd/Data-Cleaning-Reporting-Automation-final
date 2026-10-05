# Automated Data Cleaning and Reporting Dashboard

## Objective
An interactive app that cleans raw datasets on its own. It handles missing values and duplicates, finds inconsistent and invalid data, and produces data quality reports, visual summaries and business insights.

## Problem Statement
Real-world data is messy. You get blank cells, repeated rows, "M" in one row and "male" in another, prices stored as "$1,200.00", impossible ages, negative quantities and dates in mixed formats. Analysts can spend up to 80% of their time on this cleanup. This project automates that work with one reusable function, `clean_data(df)`, and shows the results in a Streamlit dashboard.

## Dataset Description
`data/raw_data.csv` is a generated Sales/Customer dataset with **550 records** (520 unique plus 30 duplicates) and these columns:

| Column | Description |
|---|---|
| CustomerID | Customer identifier (C1000–C1299) |
| CustomerName | Customer full name |
| Gender | Male / Female |
| Age | Customer age |
| Product | Product purchased (18 products) |
| Category | Electronics, Clothing, Home & Kitchen, Books, Sports |
| Region | North, South, East, West, Central |
| Quantity | Units bought |
| Price | Unit price |
| Discount | Discount as a fraction (0.10 = 10%) |
| Revenue | Quantity × Price × (1 − Discount) |
| PurchaseDate | Date of purchase (2024) |

**Problems added on purpose:** missing values (Age, Gender, Region, Price, Name, Discount, Date), 30 duplicate rows, prices like `$1,234.00`, mixed date formats (`March 05, 2024`, `2024/03/05`, `05-Mar-2024`), impossible dates (`2024-13-45`), messy text (`male`, `M`, ` FEMALE `, `north`, `ELECTRONICS`, `Electronic`, `Clothes`), invalid values (age 150 / −5, negative quantity/price, discount `15` instead of `0.15`) and outliers (quantity 250).

## Technologies Used
Python 3.9+, Pandas, NumPy, Matplotlib, Seaborn, Streamlit, OpenPyXL.

## Data Cleaning Process (`cleaning.py`)
1. **Load** the raw CSV/Excel file.
2. **Standardize column names**: `CustomerID` → `customer_id`.
3. **Remove extra spaces** and turn `N/A`, `null`, `?`, `-` into proper missing values.
4. **Convert data types**: columns with "date" in the name become dates (mixed formats accepted). Text columns that are mostly numbers become numbers, with `$`, `,` and `%` removed.
5. **Standardize text**: Title Case, one set of gender values, and fixed category/region spellings.
6. **Remove duplicates**, checked after text cleanup so ` north` and `North` count as the same.
7. **Fix invalid values**: negative quantities/prices/revenue and ages over 110 or below 0 are marked as missing. Discounts like `15` become `0.15`.
8. **Handle outliers** with the IQR rule (k = 3). Extreme values are capped, not deleted. Prices are checked per product.
9. **Handle missing values**: rows with a missing or invalid date are removed. Numbers are filled with the median (price uses the median for that product). Categories get the most common value, free text gets "Unknown" and IDs get "UNKNOWN".
10. **Recalculate revenue** so it matches the cleaned quantity, price and discount.
11. **Save** the result to `data/cleaned_data.csv`.

```python
from cleaning import clean_data, clean_data_with_report
cleaned = clean_data(raw_df)                       # just the cleaned data
cleaned, report = clean_data_with_report(raw_df)   # cleaned data + quality report
```

## Data Quality Checks
The report covers total rows and columns, missing values (per column), duplicates, invalid values (per column), outliers (per column), data types before and after, clean and removed records, cleaning percentage and a **Data Quality Score**:

```
Score = 50% × completeness + 25% × uniqueness + 25% × validity
```

| Data Quality Check | Before | After |
|---|---|---|
| Total Records | 550 | 510 |
| Missing Values | 104 | 0 |
| Duplicate Records | 30 | 0 |
| Invalid Values | 34 | 0 |

## Reporting Automation (`reporting.py`)
- Quality summary table and column profile
- Data-quality and business charts (Matplotlib/Seaborn)
- Insights calculated from the data, never hard-coded
- Downloadable **HTML summary report** with charts built in, plus an **Excel report** (Quality Summary, Column Profile, Cleaning Log, Insights, Cleaned Data)

## Dashboard Features (`app.py`)
- **Sidebar:** file upload (CSV/Excel), date range, category, region and product filters
- **KPI cards:** Total Records, Clean Records, Missing Fixed, Duplicates Removed, Invalid Fixed, Quality Score
- **Tabs:**
  1. *Overview*: summary table, quality score, cleaning steps, data preview
  2. *Data Cleaning*: raw data, missing/duplicate/invalid analysis, before vs after
  3. *Data Quality*: missing by column, duplicates, data types, outliers, completeness
  4. *Business Analysis*: revenue by category/region, top 10 products, monthly trend, quantity sold, customer distribution
  5. *Automated Report*: insights and all downloads

## Business Insights (examples, calculated live)
Highest-revenue category and region, top product by revenue and by units, monthly revenue trend (best and worst month), average order value, revenue share by gender, unique customers, and how many records were missing, duplicated or cleaned.

## Project Structure
```
data-cleaning-reporting/
├── app.py              # Streamlit dashboard
├── cleaning.py         # clean_data() pipeline + sample data generator
├── reporting.py        # charts, insights, HTML/Excel reports
├── requirements.txt
├── README.md
├── data/
│   ├── raw_data.csv
│   └── cleaned_data.csv
└── reports/            # generated reports are saved here
```

## Installation
```bash
cd data-cleaning-reporting
python -m venv venv
# Windows: venv\Scripts\activate     macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

## How to Run
```bash
streamlit run app.py
```
The dashboard opens at http://localhost:8501. To rebuild the sample data run `python cleaning.py`.

## Expected Output
- KPI cards showing 550 raw records → 510 clean records, about 104 missing values fixed, 30 duplicates removed, 34 invalid values fixed and a quality score of 100%
- Charts for data quality and business performance
- `data/cleaned_data.csv`, plus `reports/summary_report.html` and `reports/data_quality_report.xlsx`

## Using Your Own Dataset
Upload any CSV/Excel file from the sidebar. The pipeline works out column types from names and values:
- columns ending in `_id` are treated as IDs
- columns containing `date` are read as dates
- names containing `price`, `quantity`, `revenue`, `amount`, `cost` or `discount` must not be negative
- `age` must be between 0 and 110

If business columns (revenue, category, region, product, date, quantity) are missing, the app shows a warning and skips those charts instead of crashing. To add your own rules, edit `VALUE_FIXES`, `GENDER_MAP` or `NON_NEGATIVE_KEYWORDS` at the top of `cleaning.py`.
