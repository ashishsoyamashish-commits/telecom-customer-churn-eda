"""Telecom Customer Churn: preprocessing and exploratory data analysis."""

from pathlib import Path
import sys
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
DATA_URLS = [
    "https://raw.githubusercontent.com/Varijoo/telecom-churn-analysis/main/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/abdrahmansoltan/Telco-Customer-Churn/main/WA_Fn-UseC_-Telco-Customer-Churn.csv",
]

sns.set_theme(style="whitegrid")


def load_data(path: Path) -> pd.DataFrame:
    """Load the dataset and report a helpful message if it is missing."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        last_error = None
        for url in DATA_URLS:
            try:
                print(f"Dataset not found locally. Trying download from: {url}")
                request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(request, timeout=30) as response:
                    content = response.read()
                if not content.startswith(b"customerID,"):
                    raise ValueError("Downloaded file does not look like the expected CSV dataset.")
                path.write_bytes(content)
                print(f"Dataset downloaded to: {path}")
                break
            except Exception as exc:
                last_error = exc
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found and automatic download failed.\n"
                f"Save WA_Fn-UseC_-Telco-Customer-Churn.csv to: {path}\n"
                f"Download it from the Kaggle source in README.md.\n"
                f"Last download error: {last_error}"
            )
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean common issues in the IBM Telco Customer Churn dataset."""
    data = df.copy()
    data.columns = data.columns.str.strip()

    # The original file can contain blank strings in TotalCharges.
    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(
            data["TotalCharges"].replace(r"^\s*$", np.nan, regex=True),
            errors="coerce",
        )

    before = len(data)
    data = data.drop_duplicates()
    print(f"Duplicate rows removed: {before - len(data)}")

    if "TotalCharges" in data.columns and "MonthlyCharges" in data.columns:
        # Zero-tenure customers often have blank TotalCharges in the source data.
        zero_tenure = (
            data["tenure"].eq(0)
            if "tenure" in data.columns
            else pd.Series(False, index=data.index)
        )
        data.loc[
            data["TotalCharges"].isna() & zero_tenure, "TotalCharges"
        ] = 0

        remaining_missing = data["TotalCharges"].isna()
        if remaining_missing.any():
            median_total = data["TotalCharges"].median()
            data.loc[remaining_missing, "TotalCharges"] = median_total

    return data


def save_plot(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved chart: outputs/{filename}")


def make_visualizations(df: pd.DataFrame) -> None:
    """Generate charts used in the exploratory analysis."""
    # 1. Overall churn distribution
    if "Churn" in df.columns:
        plt.figure(figsize=(7, 5))
        sns.countplot(data=df, x="Churn", color="steelblue")
        plt.title("Customer Churn Distribution")
        plt.xlabel("Churn")
        plt.ylabel("Number of Customers")
        save_plot("churn_distribution.png")

    # 2. Churn rate by contract
    if {"Contract", "Churn"}.issubset(df.columns):
        contract_rate = (
            df.assign(ChurnFlag=df["Churn"].map({"Yes": 1, "No": 0}))
            .groupby("Contract", dropna=False)["ChurnFlag"]
            .mean()
            .mul(100)
            .reset_index(name="ChurnRate")
        )
        plt.figure(figsize=(8, 5))
        sns.barplot(data=contract_rate, x="Contract", y="ChurnRate", color="steelblue")
        plt.title("Churn Rate by Contract Type")
        plt.xlabel("Contract Type")
        plt.ylabel("Churn Rate (%)")
        plt.xticks(rotation=15)
        save_plot("churn_by_contract.png")

    # 3. Churn rate by internet service
    if {"InternetService", "Churn"}.issubset(df.columns):
        internet_rate = (
            df.assign(ChurnFlag=df["Churn"].map({"Yes": 1, "No": 0}))
            .groupby("InternetService", dropna=False)["ChurnFlag"]
            .mean()
            .mul(100)
            .reset_index(name="ChurnRate")
        )
        plt.figure(figsize=(8, 5))
        sns.barplot(data=internet_rate, x="InternetService", y="ChurnRate", color="steelblue")
        plt.title("Churn Rate by Internet Service")
        plt.xlabel("Internet Service")
        plt.ylabel("Churn Rate (%)")
        save_plot("churn_by_internet_service.png")

    # 4. Monthly charges by churn status
    if {"MonthlyCharges", "Churn"}.issubset(df.columns):
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x="Churn", y="MonthlyCharges", color="steelblue")
        plt.title("Monthly Charges by Churn Status")
        plt.xlabel("Churn")
        plt.ylabel("Monthly Charges")
        save_plot("monthly_charges_by_churn.png")

    # 5. Tenure distribution
    if "tenure" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(data=df, x="tenure", bins=30, kde=True, color="steelblue")
        plt.title("Customer Tenure Distribution")
        plt.xlabel("Tenure (months)")
        plt.ylabel("Number of Customers")
        save_plot("tenure_distribution.png")


def report_quality(df: pd.DataFrame) -> None:
    print("\n--- DATASET OVERVIEW ---")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]:,}")
    print("\n--- DATA TYPES ---")
    print(df.dtypes)
    print("\n--- MISSING VALUES ---")
    print(df.isna().sum().sort_values(ascending=False))
    print(f"\nDuplicate rows: {df.duplicated().sum()}")
    print("\n--- NUMERICAL SUMMARY ---")
    print(df.describe(include="all").transpose().to_string())

    if "Churn" in df.columns:
        print("\n--- CHURN COUNTS ---")
        print(df["Churn"].value_counts(dropna=False))
        print("\n--- CHURN PERCENTAGES ---")
        print((df["Churn"].value_counts(normalize=True, dropna=False) * 100).round(2))


def analyze_outliers_iqr(df: pd.DataFrame, column: str = "MonthlyCharges") -> None:
    """Report potential IQR outliers; do not automatically delete them."""
    if column not in df.columns:
        return
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    count = int(((values < lower) | (values > upper)).sum())
    print(f"\n--- IQR OUTLIER CHECK: {column} ---")
    print(f"Lower bound: {lower:.2f}; upper bound: {upper:.2f}")
    print(f"Potential outliers: {count}")
    print("Potential outliers are reported, not removed automatically.")


def main() -> int:
    try:
        raw = load_data(DATA_PATH)
    except FileNotFoundError as exc:
        print(exc)
        return 1

    print("--- RAW DATA QUALITY CHECK ---")
    report_quality(raw)

    cleaned = clean_data(raw)
    print("\n--- CLEANED DATA QUALITY CHECK ---")
    report_quality(cleaned)
    analyze_outliers_iqr(cleaned)
    make_visualizations(cleaned)

    cleaned_path = OUTPUT_DIR / "cleaned_telco_data.csv"
    cleaned.to_csv(cleaned_path, index=False)
    print(f"\nCleaned dataset saved to: {cleaned_path}")
    print("\nAnalysis complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
