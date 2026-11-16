"""
Feature engineering shared by train_model.py and app.py.

This lives in its own module on purpose. The saved model stores a reference to
add_derived_features; if that function were defined inside train_model.py, loading
the model from app.py would fail, because Python would look for it in the wrong place.
"""
import numpy as np

ANNUAL_INTEREST_RATE = 9.0  # % per year. An assumption used only for the EMI estimate.


def monthly_emi(principal_thousands, months, annual_rate=ANNUAL_INTEREST_RATE):
    """Standard EMI (equated monthly instalment) formula. Principal is in thousands."""
    principal = principal_thousands * 1000
    r = annual_rate / 12 / 100
    growth = (1 + r) ** months
    return principal * r * growth / (growth - 1)


def add_derived_features(df):
    """
    Adds three columns the model learns from:
      TotalIncome  combined applicant and co-applicant income
      EMI          estimated monthly repayment for the requested loan
      DTI          debt-to-income ratio: EMI divided by total income
    """
    out = df.copy()
    out["TotalIncome"] = out["ApplicantIncome"] + out["CoapplicantIncome"]
    out["EMI"] = monthly_emi(out["LoanAmount"], out["Loan_Amount_Term"])
    out["DTI"] = out["EMI"] / out["TotalIncome"].replace(0, np.nan)
    return out
