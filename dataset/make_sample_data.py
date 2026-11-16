"""
Creates a SYNTHETIC loan dataset with the same columns as the Kaggle
"Loan Eligibility Prediction" dataset (train.csv).

Purpose: lets the project run end-to-end before you download the real data.
The values are randomly generated, so do NOT report results from this file.
For your submission, download train.csv from Kaggle, save it as
dataset/loan_data.csv (or change DATA_PATH in train_model.py), and retrain.

Run:  python dataset/make_sample_data.py
"""
import os

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 600


def pick(options, probs):
    return rng.choice(options, size=n, p=probs)


df = pd.DataFrame({
    "Loan_ID": [f"LP{1000 + i:06d}" for i in range(n)],
    "Gender": pick(["Male", "Female"], [0.80, 0.20]),
    "Married": pick(["Yes", "No"], [0.65, 0.35]),
    "Dependents": pick(["0", "1", "2", "3+"], [0.58, 0.17, 0.16, 0.09]),
    "Education": pick(["Graduate", "Not Graduate"], [0.78, 0.22]),
    "Self_Employed": pick(["No", "Yes"], [0.86, 0.14]),
    "ApplicantIncome": rng.lognormal(mean=8.5, sigma=0.5, size=n).round().astype(int),
    "CoapplicantIncome": np.where(
        rng.random(n) < 0.40, 0, rng.lognormal(7.5, 0.8, n).round()
    ).astype(int),
    "Loan_Amount_Term": pick([360.0, 180.0, 120.0, 240.0, 84.0],
                             [0.84, 0.07, 0.04, 0.03, 0.02]),
    "Credit_History": pick([1.0, 0.0], [0.84, 0.16]),
    "Property_Area": pick(["Urban", "Semiurban", "Rural"], [0.33, 0.38, 0.29]),
})

# Loan amount is in thousands, roughly proportional to total household income
total_income = df["ApplicantIncome"] + df["CoapplicantIncome"]
df["LoanAmount"] = np.clip(
    (total_income / 1000) * rng.uniform(15, 35, n) + rng.normal(0, 15, n), 9, 700
).round()

# Hidden rule used to create the labels (this is what the model must learn)
monthly_payment = df["LoanAmount"] * 1000 / df["Loan_Amount_Term"]
burden = monthly_payment / total_income
score = (
    2.2 * (df["Credit_History"] == 1.0)
    + 0.6 * (df["Education"] == "Graduate")
    - 0.5 * (df["Self_Employed"] == "Yes")
    + 0.5 * (df["Property_Area"] == "Urban")
    + 0.3 * (df["Property_Area"] == "Semiurban")
    + 0.3 * (total_income > 7000)
    - 6.0 * np.clip(burden - 0.08, 0, None)
)
prob_approved = 1 / (1 + np.exp(-(score - 1.6)))
approved = rng.random(n) < prob_approved
df["Loan_Status"] = np.where(approved, "Y", "N")


# Realistic gaps in the data, so the imputation step has work to do
def add_missing(column, fraction):
    idx = rng.choice(n, size=int(fraction * n), replace=False)
    df.loc[idx, column] = np.nan


add_missing("Gender", 0.02)
add_missing("Married", 0.01)
add_missing("Dependents", 0.03)
add_missing("Self_Employed", 0.05)
add_missing("LoanAmount", 0.03)
add_missing("Loan_Amount_Term", 0.02)
add_missing("Credit_History", 0.07)

COLUMN_ORDER = [
    "Loan_ID", "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
    "Credit_History", "Property_Area", "Loan_Status",
]
df = df[COLUMN_ORDER]

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loan_data.csv")
df.to_csv(out_path, index=False)

print(f"Saved {len(df)} rows to {out_path}")
print("Approval rate:", round((df["Loan_Status"] == "Y").mean(), 3))
