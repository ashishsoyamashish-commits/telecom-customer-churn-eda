# Week 2: Telecom Customer Churn — Data Cleaning, EDA and Visualization

## Objective
Clean and explore the public IBM Telco Customer Churn sample dataset using Python. This project checks data quality, summarizes customer information, and creates charts to explore churn patterns.

## Dataset
- File: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Expected size: 7,043 customer records and 21 columns
- Target: `Churn` (Yes/No)
- Public source: IBM sample dataset. The script downloads it automatically if the CSV is missing and internet access is available.

## Files
- `main.py` — complete analysis script
- `requirements.txt` — required Python libraries
- `data/` — put the CSV here; auto-download is attempted if missing
- `outputs/` — created when the script runs; contains cleaned CSV, audit notes, summaries, tables and PNG charts
- `Telecom_Week2_Advanced_Analysis_Report.docx` — report template/methodology

## Run on Windows
1. Install Python and open Command Prompt or Anaconda Prompt.
2. Change directory to this project folder.
3. Run: `python -m pip install -r requirements.txt`
4. Run: `python main.py`
5. Open the `outputs` folder to view the results and charts.

If the automatic download fails, download the CSV from the IBM repository and save it as:
`data/WA_Fn-UseC_-Telco-Customer-Churn.csv`

## Main analysis
1. Dataset shape and preview
2. Data types and missing-value audit
3. Numeric conversion of `TotalCharges`
4. Duplicate check and missing-value handling
5. Descriptive statistics
6. Churn distribution
7. Churn by contract, internet service, payment method and technical support
8. Tenure and monthly-charge comparisons
9. Numeric correlation heatmap

## Interpretation note
This is exploratory analysis, not a production decision system. Relationships in the charts show association, not proof of cause. Run the script on the dataset before quoting any calculated output values in your final submission.
