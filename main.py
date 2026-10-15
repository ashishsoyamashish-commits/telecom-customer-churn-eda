"""
Week 2 - Telecom Customer Churn: Data Cleaning, EDA and Visualization
Run: python main.py
Dataset expected at data/WA_Fn-UseC_-Telco-Customer-Churn.csv
If missing, the script attempts to download the public IBM sample dataset.
"""
from pathlib import Path
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
DATA_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

DATA_FILE = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

def load_data():
    if not DATA_FILE.exists():
        print("Dataset not found. Downloading public IBM Telco sample...")
        try:
            urllib.request.urlretrieve(DATA_URL, DATA_FILE)
        except Exception as exc:
            raise SystemExit(
                "Could not download the dataset. Download the IBM Telco Customer Churn CSV "
                "and save it as: " + str(DATA_FILE) + f"\nReason: {exc}"
            )
    df = pd.read_csv(DATA_FILE)
    # Accept either common filename/content version.
    return df

def main():
    df = load_data()
    print("\\n=== TELECOM CUSTOMER CHURN: WEEK 2 EDA ===")
    print("Dataset shape:", df.shape)
    print("\\nFirst five rows:")
    print(df.head().to_string(index=False))
    print("\\nColumn data types before cleaning:")
    print(df.dtypes.to_string())

    # Standardize whitespace and convert TotalCharges to numeric.
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Audit and clean.
    duplicate_rows = int(df.duplicated().sum())
    missing_before = df.isna().sum()
    if "TotalCharges" in df.columns and "tenure" in df.columns:
        # Blank TotalCharges commonly belongs to new customers with tenure 0.
        df.loc[df["TotalCharges"].isna() & (df["tenure"] == 0), "TotalCharges"] = 0
    # Any remaining numeric missing values use median; categorical use mode.
    for col in df.select_dtypes(include="number").columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include="object").columns:
        if df[col].isna().any():
            mode = df[col].mode(dropna=True)
            df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")

    # Avoid double-counting exact duplicate rows.
    df = df.drop_duplicates().copy()

    # Save data audit and cleaned data.
    audit = [
        f"Rows after cleaning: {df.shape[0]}",
        f"Columns: {df.shape[1]}",
        f"Duplicate rows found before cleaning: {duplicate_rows}",
        "",
        "Missing values before cleaning:",
        missing_before.to_string(),
        "",
        "Missing values after cleaning:",
        df.isna().sum().to_string(),
    ]
    (OUT_DIR / "data_audit.txt").write_text("\\n".join(audit), encoding="utf-8")
    df.to_csv(OUT_DIR / "cleaned_telecom_data.csv", index=False)

    print("\\nDuplicate rows found:", duplicate_rows)
    print("\\nMissing values before cleaning:")
    print(missing_before[missing_before > 0].to_string() if (missing_before > 0).any() else "None")
    print("\\nMissing values after cleaning:", int(df.isna().sum().sum()))

    # Numeric descriptive statistics.
    numeric = df.select_dtypes(include="number")
    numeric.describe().to_csv(OUT_DIR / "numeric_summary.csv")
    print("\\nNumeric summary:")
    print(numeric.describe().round(2).to_string())

    if "Churn" not in df.columns:
        raise SystemExit("The dataset does not contain the required 'Churn' column.")

    # 1. Overall churn distribution
    churn_counts = df["Churn"].value_counts().reindex(["No", "Yes"]).fillna(0)
    churn_counts.to_csv(OUT_DIR / "churn_counts.csv", header=["customers"])
    plt.figure(figsize=(6, 4))
    churn_counts.plot(kind="bar")
    plt.title("Telecom Customer Churn Distribution")
    plt.xlabel("Churn")
    plt.ylabel("Number of customers")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "01_churn_distribution.png", dpi=160)
    plt.close()

    # 2. Churn percentage by contract type
    if "Contract" in df.columns:
        contract_rate = pd.crosstab(df["Contract"], df["Churn"], normalize="index") * 100
        contract_rate.to_csv(OUT_DIR / "churn_by_contract_percent.csv")
        ax = contract_rate.reindex(columns=["No", "Yes"]).plot(kind="bar", figsize=(8, 5))
        ax.set_title("Churn Percentage by Contract Type")
        ax.set_xlabel("Contract type")
        ax.set_ylabel("Percentage (%)")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "02_churn_by_contract.png", dpi=160)
        plt.close()

    # 3. Tenure distribution by churn
    if "tenure" in df.columns:
        groups = [df.loc[df["Churn"] == label, "tenure"].dropna() for label in ["No", "Yes"]]
        plt.figure(figsize=(7, 4))
        plt.boxplot(groups, tick_labels=["No churn", "Churn"])
        plt.title("Customer Tenure by Churn Status")
        plt.xlabel("Churn status")
        plt.ylabel("Tenure (months)")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "03_tenure_by_churn.png", dpi=160)
        plt.close()

    # 4. Monthly charges by churn
    if "MonthlyCharges" in df.columns:
        groups = [df.loc[df["Churn"] == label, "MonthlyCharges"].dropna() for label in ["No", "Yes"]]
        plt.figure(figsize=(7, 4))
        plt.boxplot(groups, tick_labels=["No churn", "Churn"])
        plt.title("Monthly Charges by Churn Status")
        plt.xlabel("Churn status")
        plt.ylabel("Monthly charges")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "04_monthly_charges_by_churn.png", dpi=160)
        plt.close()

    # 5. Correlation heatmap (matplotlib only, no extra plotting dependency)
    corr = numeric.corr(numeric_only=True)
    corr.to_csv(OUT_DIR / "numeric_correlation.csv")
    if not corr.empty:
        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(corr, aspect="auto", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
        ax.set_yticklabels(corr.columns, fontsize=8)
        ax.set_title("Correlation Heatmap of Numeric Variables")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "05_correlation_heatmap.png", dpi=160)
        plt.close(fig)

    # Churn rates by selected categorical variables.
    for col in ["InternetService", "PaymentMethod", "SeniorCitizen", "TechSupport"]:
        if col in df.columns:
            pd.crosstab(df[col], df["Churn"], normalize="index").mul(100).to_csv(
                OUT_DIR / f"churn_by_{col.lower()}_percent.csv"
            )

    print("\\nChurn counts:")
    print(churn_counts.to_string())
    print("\\nAnalysis completed successfully.")
    print("Cleaned data, tables, audit notes and charts are saved in:", OUT_DIR)

if __name__ == "__main__":
    main()
