# Loan Approval Predictor (Roll No. 22)

**Project 3 from the individual ML project list. ML type: supervised classification (binary).**

The project has two parts:

1. **Model development** (`train_model.py`): explores the data, builds and compares models, and saves the best one.
2. **Web application** (`app.py`): an advanced loan assessment tool. It estimates approval, repayment and debt ratio, saves every assessment to a history database, and lets you search and manage that history.

> **Important:** the included `dataset/loan_data.csv` is **synthetic**. It has the same columns as the Kaggle "Loan Eligibility Prediction" dataset, and it exists so the project runs before you download the real data. Replace it with the real file before you write your report. All numbers in this README come from the synthetic sample and will change with real data.

---

## 1. Project structure

```
loan_approval_project/
├── app.py                    # Flask web application (routes, validation, prediction)
├── train_model.py            # Parts B to L: EDA, preprocessing, training, evaluation, saving
├── features.py               # shared feature engineering (income, EMI, debt-to-income ratio)
├── database.py               # SQLite storage for the assessment history
├── dataset/
│   ├── make_sample_data.py   # creates the synthetic sample (not needed with real data)
│   └── loan_data.csv         # synthetic sample; replace with the Kaggle train.csv
├── templates/
│   ├── base.html             # layout, navigation, loading overlay container
│   ├── index.html            # new assessment form
│   ├── result.html           # decision report
│   ├── history.html          # history with search, filters and delete
│   └── message.html          # simple message page (e.g. not found)
├── static/
│   ├── style.css             # theme and layout
│   └── app.js                # loading sequence and delete confirmation
├── loan_model.joblib         # created by train_model.py: the saved model pipeline
├── loan_history.db           # created by app.py: the assessment history
├── outputs/                  # created by train_model.py: plots and metric tables
├── requirements.txt
└── README.md
```

## 2. Setup and run

Use a terminal opened in the project folder. On Windows, open the folder in File Explorer, click the address bar, type `cmd`, and press Enter.

```bash
# 1. (Recommended) create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 2. Install the libraries
python -m pip install -r requirements.txt

# 3. (Optional) use the real dataset
#    Download train.csv from the Kaggle "Loan Eligibility Prediction" dataset,
#    save it as dataset/loan_data.csv (overwrite the sample), then continue.

# 4. Train the model (creates loan_model.joblib and the outputs/ folder)
python train_model.py

# 5. Start the web application (keep this window open while you use the app)
python app.py
#    Open http://127.0.0.1:5000 in your browser
```

If `python app.py` stops immediately, it is almost always because step 4 has not been run yet (the model file is missing). The app shows a clear message in the browser if the model cannot be loaded.

## 3. Using the application

- **New assessment:** fill in the three sections (applicant profile, income and loan request, credit profile) and press **Run assessment**. A short loading sequence appears, then the decision report opens.
- **Decision report:** shows Approved or Not approved, the probability of approval, the estimated monthly repayment (EMI), total income, the debt-to-income ratio with a rating, and every input. It has print support.
- **History:** every assessment is saved in `loan_history.db`. The page shows totals and approval rate, lets you search by applicant name or reference (for example `LA-1A2B3C4D`), filter by decision, open any report, and delete a record after confirmation.
- **Refreshing a report** does not resubmit the form, because the app redirects to the report after each assessment.

## 4. Problem definition (Part A)

| Item | Answer |
|---|---|
| Aim | Predict whether a loan application will be approved from applicant information |
| Input variables | Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area |
| Derived variables | TotalIncome, EMI (estimated monthly repayment), DTI (EMI divided by total income) |
| Target variable | `Loan_Status`: Y = approved (encoded 1), N = not approved (encoded 0) |
| ML type | Supervised, binary classification |
| Why classification | The output is one of two categories, not a number. A regression model predicts a value, which does not fit a yes/no decision. |

## 5. Dataset understanding (Part B)

Run by `inspect()` in `train_model.py`. It prints the first five records, dimensions, data types, missing values, duplicate count, feature lists and class balance.

- **Missing values** appear in Gender, Married, Dependents, Self_Employed, LoanAmount, Loan_Amount_Term and Credit_History. Credit_History has the most gaps.
- **Numerical features:** ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History (coded 0/1)
- **Categorical features:** Gender, Married, Dependents, Education, Self_Employed, Property_Area
- **Class balance:** about two-thirds of applications are approved, so accuracy alone is misleading. A model that approves everyone scores about 68% accuracy.

## 6. Preprocessing (Part C) and why

| Decision | Reason |
|---|---|
| Drop `Loan_ID` | It is an identifier with no information about the applicant |
| Remove duplicate rows | Duplicates would let the same applicant appear in both training and test sets |
| Remove rows with no label | Cannot train or evaluate without the outcome |
| Derived features (`features.py`) | Income alone is a weak signal. Repayment size relative to income is what lenders check, so the model learns from TotalIncome, EMI and DTI as well |
| Median fill for numeric gaps | The median is not pulled by very large incomes (income is right-skewed) |
| Most-frequent fill for categories and Credit_History | The most common value is a reasonable guess for a missing category |
| One-hot encoding for categories | Turns text categories into 0/1 columns without implying an order |
| Scaling for Logistic Regression and KNN only | These models compare distances or weighted sums, so features on different scales distort them. Tree models split on thresholds and do not need scaling. |
| Fills and encoding inside the pipeline | Fill values come from training data only, so the test set does not leak into training |
| Outliers | Income and loan amount have long right tails. Boxplots hide them for readability, but they stay in training. Tree models cope well with them. |

**Why the feature code is in `features.py`:** the saved pipeline stores a reference to `add_derived_features`. If that function lived in `train_model.py`, loading the model from `app.py` would fail, because Python would look for it in the wrong module.

## 7. EDA (Part D): plots and observations

Plots are saved as `outputs/eda_plots.png`: class distribution, approval rate by Credit_History, income by outcome, and approval rate by Property_Area.

1. **Credit history is the strongest pattern.** Applicants with credit history were approved about 74% of the time, against about 30% without it.
2. **Approval rates are similar across areas,** slightly higher for Urban (about 70%) than Rural (about 64%).
3. **Income alone does not separate the groups.** Median income is about the same for approved and rejected applicants (about 5,000 each in the sample).
4. **The classes are imbalanced,** with about 68% approved, which is why recall and F1 are used instead of accuracy alone.

## 8. Features and target (Part E)

- `X` = `RAW_FEATURES` in `train_model.py`: the 11 applicant inputs. The derived columns are computed inside the pipeline, so the app only sends the raw inputs.
- `y` = `Loan_Status` encoded as 1/0.
- **Derived features:**
  - `TotalIncome` = applicant income + co-applicant income
  - `EMI` = the standard monthly repayment for the loan amount and term, at an assumed 9% annual rate
  - `DTI` = EMI divided by TotalIncome
- **Recorded but not used by the model:** applicant name, age and loan purpose. The name is used for search. Age and purpose are stored with each record for reference. The form labels these fields clearly.

## 9. Train-test split (Part F)

`train_test_split(test_size=0.20, stratify=y, random_state=42)`

- 80% for training and 20% for testing, as the assignment requires.
- `stratify=y` keeps the approved/rejected ratio the same in both sets. Without it, a random split could leave very few rejections in the test set.
- `random_state=42` makes the split repeatable.

## 10. Models (Parts G and H)

| Model | Why it was included |
|---|---|
| Logistic Regression | A transparent linear model. Standard for credit scorecards. Gives smooth probabilities. |
| Decision Tree | Easy to explain as rules; a useful non-linear comparison |
| Decision Tree (tuned) | The tree after Experiment 1 |
| KNN (k=7) | A different approach (similar applicants have similar outcomes) |
| Random Forest | An ensemble of trees; often strong on tabular data |

Each model is a pipeline (derived features, preprocessing, classifier). Each is scored with 5-fold cross-validation on the training data, then fit on the full training set and tested.

## 11. Evaluation (Part I) and model selection

Metrics:
- **Accuracy:** share of correct predictions. Misleading here because of the class imbalance.
- **Precision (approved):** of applicants predicted approved, how many were approved.
- **Recall (approved):** of applicants who were approved, how many the model found.
- **Recall (not approved):** of applicants who should be declined, how many the model caught. This matters most for a lender: a missed rejection means a risky loan is approved.
- **F1 (approved):** balance of precision and recall for the approved class.
- **ROC AUC:** how well the probabilities rank applicants, regardless of the cut-off (0.5 = guessing, 1.0 = perfect).

**Selection rule** (uses only training data; the test set is never used to choose):
1. Rank all models by **5-fold cross-validated macro-F1**. Macro-F1 averages F1 over both classes, so it does not reward a model for approving everyone.
2. The best overall model is reported. The model used in the app is the best **scorecard-style** model. Scorecard-style means a model whose probability responds smoothly to every input. In this project that is Logistic Regression.

**Why the app does not use the top-ranked tree:** the tuned tree has the highest macro-F1 (0.648), but its probabilities take only a handful of distinct values. Several input changes leave the probability unchanged, so the app would appear to ignore them. Logistic Regression is the standard choice for credit scorecards in banking, and its probability responds to every input. The trade-off is a lower cross-validated score (0.609 vs 0.648), which the table below makes visible.

Results on the synthetic sample (120-row test set):

| Model | CV macro-F1 (train) | CV ROC AUC (train) | Accuracy | Precision (approved) | Recall (approved) | Recall (not approved) | F1 (approved) | ROC AUC |
|---|---|---|---|---|---|---|---|---|
| Decision Tree (tuned) | 0.648 | 0.638 | 0.717 | 0.805 | 0.765 | 0.615 | 0.785 | 0.746 |
| **Logistic Regression (used in the app)** | 0.609 | 0.686 | 0.742 | 0.772 | 0.877 | 0.462 | 0.821 | 0.779 |
| Random Forest | 0.569 | 0.630 | 0.708 | 0.740 | 0.877 | 0.359 | 0.802 | 0.715 |
| Decision Tree | 0.553 | 0.581 | 0.758 | 0.795 | 0.864 | 0.538 | 0.828 | 0.722 |
| KNN (k=7) | 0.487 | 0.538 | 0.683 | 0.697 | 0.938 | 0.154 | 0.800 | 0.607 |

Full table: `outputs/model_comparison.csv`.

Logistic Regression's final test classification report: precision 0.64 and recall 0.46 for not approved; precision 0.77 and recall 0.88 for approved; overall accuracy 0.74.

## 12. Visualization (Part J)

- `outputs/confusion_matrix.png`: where the final model's errors are
- `outputs/model_comparison.png`: F1 and recall for each model
- `outputs/feature_importance.png`: which inputs the final model relies on most

## 13. Model improvement (Part K): experiments

Each experiment is recorded as **Change → Result → Observation → Conclusion**.

### Experiment 1: tune the Decision Tree

- **Change:** `GridSearchCV` over `max_depth` (2, 3, 4, 5, 6, 8, unlimited) and `min_samples_leaf` (1, 5, 10, 20), scored by 5-fold macro-F1. Best: unlimited depth, `min_samples_leaf=20`.
- **Result:** cross-validated macro-F1 rose from 0.553 (default: depth 4, leaf 1) to 0.648. On the test set, recall of rejections rose from 0.538 to 0.615 and ROC AUC rose from 0.722 to 0.746, while accuracy fell from 0.758 to 0.717.
- **Observation:** a larger minimum leaf size stops the tree from memorizing individual applicants. It catches more rejections but accepts slightly more risky applications.
- **Conclusion:** the tuned tree is better on the measures that matter for a lender. The gain is modest and the test set is small (120 rows), so confirm it on real data.

### Experiment 2: class weights (`class_weight='balanced'`)

- **Change:** train Logistic Regression and Random Forest with `class_weight='balanced'`, which makes mistakes on declined applications count more.
- **Result (Logistic Regression):** recall of rejections rose from 0.462 to 0.744. Recall of approvals fell from 0.877 to 0.704, and accuracy fell from 0.742 to 0.717. Cross-validated macro-F1 rose from 0.609 to 0.636.
- **Result (Random Forest):** recall of rejections rose from 0.359 to 0.564. F1 for approvals fell from 0.802 to 0.783, and accuracy stayed at 0.708.
- **Observation:** balanced weights catch many more rejections but also decline more good applicants. The effect is large for Logistic Regression.
- **Conclusion:** the right setting depends on the lender's priorities. If a missed bad loan costs more than a declined good applicant, balanced weights are worth it. The app keeps the default setting so the probability keeps its usual meaning. Full table: `outputs/experiment2_class_weight.csv`.

## 14. Final interpretation (Part L)

1. **Which model did you finally select?** Logistic Regression, fitted on the training data inside the preprocessing pipeline and saved for the app.
2. **Why?** It is transparent, standard for credit scorecards, and its probability responds to every applicant input. The top-ranked tree (macro-F1 0.648) is only slightly better on cross-validation, and its coarse probabilities would hide the effect of many inputs in the app.
3. **Important features?** Credit history is by far the most influential input. Next come the derived EMI, the number of dependents, education, self-employment, total income and loan amount. Coefficient magnitudes on scaled inputs show influence, not direction. Direction is seen in the app: more credit history and higher income raise the probability, while a larger loan lowers it. A shorter term also lowers it, because it raises the monthly repayment (EMI).
4. **Limitations:**
   - The data is **synthetic**, so these results describe the code, not real lending.
   - The test set has only 120 rows, so small differences are not reliable.
   - Logistic Regression assumes a straight-line effect of each input, so it cannot capture interactions between them.
   - The EMI uses an assumed 9% rate, and the debt-ratio bands are rules of thumb, not bank policy.
   - Age, loan purpose and applicant name do not feed the model. Fairness checks on gender and property area are needed before any real use.
5. **Future improvements:**
   - Train on the real Kaggle dataset and report the new numbers
   - Try a log transform on income and an interaction term, then compare
   - Tune the decision threshold to meet a target recall for rejections
   - Use nested cross-validation for a less optimistic estimate
   - Add a proper fairness audit before any real use

## 15. Front end (Parts 13 to 17)

- **Part 13, build:** `app.py` defines the routes. Templates use a shared layout (`base.html`), the form (`index.html`), the report (`result.html`) and the history (`history.html`). The styling is in `static/style.css`.
- **Part 14, load:** `joblib.load()` runs once when the app starts. Predictions never retrain the model. If the file is missing or cannot be loaded, the app shows a clear message instead of crashing.
- **Part 15, input:** the form has three sections and validates each value on the server (`parse_form()`). Choices are dropdowns, so loan terms and categories cannot be mistyped.
- **Part 16, predict:** `assess()` builds one row with the raw inputs and calls `predict()` and `predict_proba()`. The pipeline computes the derived features itself.
- **Part 17, display:** the report shows the decision, a probability gauge, the repayment picture and every input. Each assessment is stored in `loan_history.db` (`database.py`).

**What the loading sequence is:** after you press Run assessment, a short four-step progress overlay appears for about 2.8 seconds, then the form is submitted. The model itself answers in a fraction of a second. The pause is deliberate, to show that the application is working.

**Live experiment (external examiner):** start with the default example, which is about 91% approved. Lower the income, raise the loan amount, change the term to 120 months, or set credit history to "No credit history." Each change moves the probability. Then show the History page, search for the applicant's name, and open the saved report.

## 16. Changes from the previous version

- **Model responds to every input.** The earlier tree used only two inputs, so many changes had no effect. The app now uses Logistic Regression, and the selection rule is stated in the code and above.
- **Derived repayment features** (TotalIncome, EMI, DTI) make the model more realistic.
- **History:** every assessment is saved, with search by name or reference, decision filter, statistics, and delete.
- **Loading sequence** after pressing Run assessment.
- **Post-redirect-get:** refreshing a report no longer resubmits the form.
- **Safer startup:** clear error messages if the model is missing or fails to load; debug mode is off, so the server does not restart when files change.
- **Accurate footer:** the footer names the dataset actually used (synthetic) instead of calling it a student dataset.
- **Dropdowns** for loan term and categories instead of free text.
- **Shared feature code** moved to `features.py`, so the saved model loads correctly from the app.

## 17. Viva preparation

Be ready to explain:

- **Why classification and not regression?** The output is a yes/no category.
- **Why stratified splitting?** It keeps the approved/rejected ratio the same in both sets.
- **Why not accuracy alone?** About 68% of applications are approved, so a model that approves everyone scores about 68%.
- **What is recall (not approved), and why does it matter?** It is the share of bad applications the model catches. Missing one means approving a risky loan.
- **Why scale for Logistic Regression and KNN but not for trees?** Scaling affects distances and weighted sums; tree splits depend only on the order of values.
- **Why is the pipeline saved instead of only the model?** The pipeline keeps the derived features, imputation and encoding together, so new applications are prepared exactly as the training data was.
- **Why is the feature code in its own file?** The saved model refers to it by module name, so it must be importable by both `train_model.py` and `app.py`.
- **Why does the app not retrain?** Training takes time, and the saved model is what the app should use. Retraining happens only when the data changes.
- **Why Logistic Regression if the tree scored higher?** The tree's probabilities take few distinct values, so the app would appear to ignore many inputs. Logistic Regression is the standard credit-scorecard model, and it responds smoothly to every input. The trade-off is a slightly lower cross-validated score, which is shown in the table.
- **What does the EMI mean here?** The standard monthly repayment at an assumed 9% annual rate. It is an estimate for the demo, not a bank quote.
- **What would you do with the real data?** Re-run the pipeline, update the tables in this README, and re-check the experiments.

## 18. Submission checklist

- [ ] Real Kaggle dataset used (or the synthetic data clearly labelled)
- [ ] `python train_model.py` runs without errors
- [ ] `python app.py` runs; a new assessment shows the loading sequence and then a report
- [ ] Changing income, loan amount, term or credit history changes the probability
- [ ] History search, filter and delete work
- [ ] All plots in `outputs/` reviewed and explained
- [ ] Experiments written up in Change → Result → Observation → Conclusion
- [ ] Part L questions answered with your own numbers
- [ ] Code, dataset instructions and this README uploaded to GitHub
- [ ] Ready to explain every stage at the viva
