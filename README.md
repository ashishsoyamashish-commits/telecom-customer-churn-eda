# Telecom Customer Churn — Data Preprocessing & EDA

A Python project for the Yuva Intern Week 1 task: **Data Preprocessing and Exploratory Data Analysis in Telecom**.

## Project overview
This project explores the IBM Telco Customer Churn sample dataset (7,043 customer records) to inspect data quality, clean data, analyze customer churn, and generate visual summaries.

### Tools and libraries
- Python 3.9+
- pandas — loading, cleaning, and summarizing tabular data
- NumPy — numerical operations
- Matplotlib and Seaborn — charts and visual analysis

## Dataset
Use the publicly available **IBM Telco Customer Churn** sample dataset. One commonly used source is Kaggle:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` and place it inside the `data/` folder. The dataset is not included in this repository.

Expected path:
`data/WA_Fn-UseC_-Telco-Customer-Churn.csv`

## Setup and run

1. Install Python 3.9 or newer.
2. Open a terminal in the project folder.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. The script automatically tries to download the CSV into `data/` on first run. If your network blocks automatic download, download it from the Kaggle link above and save it as `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`.
5. Run:

   ```bash
   python main.py
   ```

The script prints dataset summaries and saves charts in the `outputs/` folder.

## Workflow
1. Load the CSV with Pandas.
2. Inspect shape, columns, data types, summary statistics, missing values, and duplicates.
3. Convert `TotalCharges` to numeric, handling blank strings as missing values.
4. Remove duplicate rows if any are present.
5. Fill missing `TotalCharges` values using `MonthlyCharges` for zero-tenure customers when appropriate; otherwise use the median.
6. Analyze churn by contract, internet service, payment method, tenure, and monthly charges.
7. Detect potential outliers in `MonthlyCharges` using the IQR method.
8. Save visualizations and a cleaned CSV.

## Output files
The script creates:
- `outputs/churn_distribution.png`
- `outputs/churn_by_contract.png`
- `outputs/churn_by_internet_service.png`
- `outputs/monthly_charges_by_churn.png`
- `outputs/tenure_distribution.png`
- `outputs/cleaned_telco_data.csv`

## Important notes
- This is exploratory analysis, not a causal study. Associations do not prove that a feature causes churn.
- The script expects the original IBM sample dataset columns.
- Do not upload private or personally identifiable customer data.

## Suggested repository description
`Python project for cleaning and exploratory analysis of the IBM Telco Customer Churn dataset using Pandas, NumPy, Matplotlib, and Seaborn.`
