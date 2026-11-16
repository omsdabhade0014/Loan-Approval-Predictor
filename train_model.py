"""
Loan Approval Prediction - model development pipeline.

Covers the assignment parts:
  B  Dataset understanding      -> inspect()
  C  Preprocessing              -> clean() + build_preprocessor()
  D  EDA (4 plots)              -> make_eda_plots()
  E  Feature / target selection -> RAW_FEATURES, TARGET
  F  80/20 stratified split     -> train_test_split()
  G  Model selection            -> candidate_models()
  H  Training                   -> score_model()
  I  Evaluation                 -> comparison table
  J  Visualization              -> confusion matrix, comparison chart, feature importance
  K  Improvement experiments    -> tune_decision_tree(), class_weight_experiment()
  L  Final model                -> chosen on TRAINING data only, then saved

Run:  python train_model.py
"""
import os

import joblib
import matplotlib

matplotlib.use("Agg")  # save plots to files; no display window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from features import add_derived_features

# ---------------------------------------------------------------- settings
DATA_PATH = "dataset/loan_data.csv"  # replace with your Kaggle train.csv if you prefer
DATASET_LABEL = "Synthetic sample dataset (replace with the Kaggle file for real results)"
MODEL_PATH = "loan_model.joblib"
OUT_DIR = "outputs"
RANDOM_STATE = 42
TARGET = "Loan_Status"  # Y = approved, N = not approved
ID_COL = "Loan_ID"

CATEGORICAL = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
RAW_NUMERIC = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]
CREDIT = ["Credit_History"]  # coded 0/1, so it is kept as a number, not scaled
DERIVED = ["TotalIncome", "EMI", "DTI"]  # computed by features.add_derived_features
NUMERIC = RAW_NUMERIC + DERIVED

# Part E: X is the raw applicant inputs. The derived columns are computed inside the pipeline,
# so the app only needs to send the raw inputs.
RAW_FEATURES = CATEGORICAL + RAW_NUMERIC + CREDIT


# ------------------------------------------------------- Part B: inspection
def inspect(raw):
    print("=" * 70)
    print("PART B - DATASET UNDERSTANDING")
    print("=" * 70)
    print("First five records:\n", raw.head(), "\n")
    print("Dimensions (rows, columns):", raw.shape, "\n")
    print("Data types:\n", raw.dtypes, "\n")
    print("Missing values per column:\n", raw.isnull().sum(), "\n")
    print("Duplicate records:", raw.duplicated().sum(), "\n")
    print("Numerical features:", RAW_NUMERIC + CREDIT)
    print("Categorical features:", CATEGORICAL, "\n")
    print("Class balance (Y = approved, N = not approved):\n",
          raw[TARGET].value_counts(), "\n")


# ------------------------------------------------ Part C: preprocessing
def clean(raw):
    """Remove the ID column, duplicate rows and rows with no label; encode the target."""
    df = raw.drop(columns=[ID_COL], errors="ignore").drop_duplicates()
    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].map({"Y": 1, "N": 0}).astype(int)
    return df


def build_preprocessor(scale_numeric):
    """
    Missing values are filled inside the pipeline, so the fill values learned from
    training data are the ones applied to new applications.
    Scaling is needed for distance- or weight-based models (Logistic Regression, KNN)
    but not for tree models (Decision Tree, Random Forest).
    """
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))

    return ColumnTransformer([
        ("num", Pipeline(numeric_steps), NUMERIC),
        ("credit", SimpleImputer(strategy="most_frequent"), CREDIT),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL),
    ])


def make_pipeline(classifier, scale_numeric):
    """Full pipeline: derived features -> preprocessing -> classifier."""
    return Pipeline([
        ("features", FunctionTransformer(add_derived_features)),
        ("prep", build_preprocessor(scale_numeric)),
        ("clf", classifier),
    ])


# ---------------------------------------------------- Part D: EDA plots
def make_eda_plots(df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Exploratory Data Analysis", fontsize=14)

    # 1. Class distribution
    counts = df[TARGET].map({1: "Approved", 0: "Not approved"}).value_counts()
    axes[0, 0].bar(counts.index, counts.values, color=["#4C78A8", "#E45756"])
    axes[0, 0].set_title("Class distribution")
    axes[0, 0].set_ylabel("Applications")

    # 2. Approval rate by credit history
    credit_rate = df.groupby("Credit_History")[TARGET].mean()
    axes[0, 1].bar([str(int(v)) for v in credit_rate.index], credit_rate.values,
                   color="#72B7B2")
    axes[0, 1].set_title("Approval rate by Credit History (1 = has history)")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_ylabel("Approval rate")

    # 3. Applicant income by outcome
    income_no = df.loc[df[TARGET] == 0, "ApplicantIncome"].dropna()
    income_yes = df.loc[df[TARGET] == 1, "ApplicantIncome"].dropna()
    axes[1, 0].boxplot([income_no, income_yes], showfliers=False)
    axes[1, 0].set_xticks([1, 2])
    axes[1, 0].set_xticklabels(["Not approved", "Approved"])
    axes[1, 0].set_title("Applicant income by outcome (outliers hidden)")
    axes[1, 0].set_ylabel("ApplicantIncome")

    # 4. Approval rate by property area
    area_rate = df.groupby("Property_Area")[TARGET].mean().sort_values(ascending=False)
    axes[1, 1].bar(area_rate.index, area_rate.values, color="#F58518")
    axes[1, 1].set_title("Approval rate by Property Area")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_ylabel("Approval rate")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "eda_plots.png"), dpi=120)
    plt.close(fig)

    print("=" * 70)
    print("PART D - EDA OBSERVATIONS (use these numbers in your report)")
    print("=" * 70)
    print("Approval rate by Credit_History:\n", credit_rate.round(3), "\n")
    print("Approval rate by Property_Area:\n", area_rate.round(3), "\n")
    print("Median ApplicantIncome - not approved:", income_no.median(),
          "| approved:", income_yes.median(), "\n")


# ------------------------------------------- Parts G, H, I: models
def candidate_models():
    return {
        "Logistic Regression": make_pipeline(
            LogisticRegression(max_iter=1000), scale_numeric=True),
        "Decision Tree": make_pipeline(
            DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE), scale_numeric=False),
        "KNN (k=7)": make_pipeline(
            KNeighborsClassifier(n_neighbors=7), scale_numeric=True),
        "Random Forest": make_pipeline(
            RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE), scale_numeric=False),
    }


SELECTION_METRIC = "f1_macro"  # averages F1 over both classes, so rejections count too
# Scorecard-style models produce smooth probabilities that respond to every input.
# Logistic Regression is the standard choice for credit scorecards in banking, so the
# app's final model is picked from this list. The overall best model is still reported.
SCORECARD_MODELS = ["Logistic Regression"]


def score_model(name, pipe, X_train, y_train, X_test, y_test):
    """Cross-validated scores on training data (used for selection) + test-set metrics."""
    cv = cross_validate(pipe, X_train, y_train, cv=5,
                        scoring={"macro_f1": SELECTION_METRIC, "roc_auc": "roc_auc"})
    cv_f1 = cv["test_macro_f1"].mean()
    cv_auc = cv["test_roc_auc"].mean()
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    row = {
        "Model": name,
        "CV macro-F1 (train)": round(cv_f1, 3),
        "CV ROC AUC (train)": round(cv_auc, 3),
        "Accuracy": round(accuracy_score(y_test, y_pred), 3),
        "Precision (approved)": round(precision_score(y_test, y_pred, zero_division=0), 3),
        "Recall (approved)": round(recall_score(y_test, y_pred), 3),
        "Recall (not approved)": round(recall_score(y_test, y_pred, pos_label=0), 3),
        "F1 (approved)": round(f1_score(y_test, y_pred), 3),
        "ROC AUC": round(roc_auc_score(y_test, y_prob), 3),
    }
    return row, pipe


# ------------------------------------ Part K: improvement experiments
def tune_decision_tree(X_train, y_train):
    base = make_pipeline(DecisionTreeClassifier(random_state=RANDOM_STATE), scale_numeric=False)
    grid = {
        "clf__max_depth": [2, 3, 4, 5, 6, 8, None],
        "clf__min_samples_leaf": [1, 5, 10, 20],
    }
    search = GridSearchCV(base, grid, cv=5, scoring=SELECTION_METRIC, n_jobs=-1)
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def class_weight_experiment(X_train, y_train, X_test, y_test):
    """Experiment 2: does class_weight='balanced' help catch rejections?"""
    rows = []
    for label, weight in [("default", None), ("balanced", "balanced")]:
        for name, clf in [
            ("Logistic Regression", LogisticRegression(max_iter=1000, class_weight=weight)),
            ("Random Forest", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE,
                                                     class_weight=weight)),
        ]:
            pipe = make_pipeline(clf, scale_numeric=(name == "Logistic Regression"))
            row, _ = score_model(f"{name} ({label} weights)", pipe,
                                 X_train, y_train, X_test, y_test)
            rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------- Part J: visualizations
def plot_confusion(pipe, X_test, y_test, name):
    cm = confusion_matrix(y_test, pipe.predict(X_test))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Not approved", "Approved"])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion matrix - {name}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=120)
    plt.close(fig)


def plot_comparison(table):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(table))
    width = 0.38
    ax.bar(x - width / 2, table["F1 (approved)"], width, label="F1 (approved)", color="#4C78A8")
    ax.bar(x + width / 2, table["Recall (not approved)"], width,
           label="Recall (not approved)", color="#E45756")
    ax.set_xticks(x)
    ax.set_xticklabels(table["Model"], rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Model comparison on the test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "model_comparison.png"), dpi=120)
    plt.close(fig)


def plot_feature_importance(pipe, name):
    """Shows which inputs the final model relies on most (tree models and Logistic Regression)."""
    prep = pipe.named_steps["prep"]
    clf = pipe.named_steps["clf"]
    names = prep.get_feature_names_out()
    if hasattr(clf, "feature_importances_"):
        importance = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importance = np.abs(clf.coef_[0])
    else:
        print(f"\n{name} has no feature importance; skipping that plot.")
        return

    top = pd.Series(importance, index=names).sort_values(ascending=True).tail(10)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top.index, top.values, color="#54A24B")
    ax.set_title(f"Top features - {name}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "feature_importance.png"), dpi=120)
    plt.close(fig)
    print(f"\nMost important features ({name}):\n", top.sort_values(ascending=False).round(3))


# ------------------------------------------------------------- main
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    raw = pd.read_csv(DATA_PATH)
    inspect(raw)

    df = clean(raw)
    make_eda_plots(df)

    X = df[RAW_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    print("=" * 70)
    print("PART F - SPLIT")
    print("=" * 70)
    print(f"Training rows: {len(X_train)} | Testing rows: {len(X_test)}")
    print("Approval rate train:", round(y_train.mean(), 3),
          "| test:", round(y_test.mean(), 3), "\n")

    # Part G / H / I: train and compare the base models
    results, fitted = [], {}
    for name, pipe in candidate_models().items():
        row, fitted_pipe = score_model(name, pipe, X_train, y_train, X_test, y_test)
        results.append(row)
        fitted[name] = fitted_pipe

    # Part K, experiment 1: tune the Decision Tree and keep it as a candidate
    tuned, best_params = tune_decision_tree(X_train, y_train)
    print("=" * 70)
    print("PART K - EXPERIMENT 1: Decision Tree tuning (GridSearchCV, 5-fold, macro-F1)")
    print("=" * 70)
    print("Best parameters:", best_params, "\n")
    row, fitted_tuned = score_model("Decision Tree (tuned)", tuned,
                                    X_train, y_train, X_test, y_test)
    results.append(row)
    fitted["Decision Tree (tuned)"] = fitted_tuned

    table = pd.DataFrame(results).sort_values("CV macro-F1 (train)", ascending=False)
    print("=" * 70)
    print("PART I - MODEL COMPARISON (test set; selection used CV macro-F1 on training data)")
    print("=" * 70)
    print(table.to_string(index=False), "\n")
    table.to_csv(os.path.join(OUT_DIR, "model_comparison.csv"), index=False)

    # Part K, experiment 2: class weights
    exp2 = class_weight_experiment(X_train, y_train, X_test, y_test)
    print("=" * 70)
    print("PART K - EXPERIMENT 2: class_weight='balanced' vs default")
    print("=" * 70)
    print(exp2.to_string(index=False), "\n")
    exp2.to_csv(os.path.join(OUT_DIR, "experiment2_class_weight.csv"), index=False)

    # Part L: final model chosen using TRAINING-data cross-validation only (test set not used).
    # The overall best model is reported first. The app's model is the best scorecard-style
    # model (see SCORECARD_MODELS), because its probabilities respond to every input.
    overall_best = table.iloc[0]["Model"]
    scorecard_table = table[table["Model"].isin(SCORECARD_MODELS)]
    best_name = scorecard_table.iloc[0]["Model"]
    best_pipe = fitted[best_name]
    print("=" * 70)
    print("PART L - FINAL MODEL")
    print("=" * 70)
    print(f"Best overall by CV macro-F1: {overall_best}")
    print(f"Scorecard-style models considered: {SCORECARD_MODELS}")
    print(f"Selected for the app: {best_name}\n")
    print("Classification report on the test set:")
    print(classification_report(y_test, best_pipe.predict(X_test),
                                target_names=["Not approved", "Approved"]))

    plot_confusion(best_pipe, X_test, y_test, best_name)
    plot_comparison(table)
    plot_feature_importance(best_pipe, best_name)

    joblib.dump({
        "model": best_pipe,
        "features": RAW_FEATURES,
        "model_name": best_name,
        "dataset_label": DATASET_LABEL,
    }, MODEL_PATH)
    print(f"\nModel saved successfully to {MODEL_PATH}")
    print(f"Plots and tables saved in the '{OUT_DIR}' folder.")


if __name__ == "__main__":
    main()
