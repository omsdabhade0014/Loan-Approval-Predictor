"""
Loan Approval Predictor - web application (Flask).

Parts 13-17 of the pipeline:
  13 Build front end   -> templates/ + static/ + the routes below
  14 Load model        -> joblib.load() once at start-up (the model is never retrained here)
  15 Take user input   -> form, validated server-side in parse_form()
  16 Predict           -> assess() runs predict() and predict_proba() through the pipeline
  17 Display output    -> result page; every assessment is saved to loan_history.db

Run:  python app.py   then open http://127.0.0.1:5000
"""
import os
import uuid
from datetime import datetime

import joblib
import pandas as pd
from flask import Flask, abort, redirect, render_template, request, url_for

import database
from features import add_derived_features  # noqa: F401  (must be importable to load the pipeline)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "loan_model.joblib")

# ---------------------------------------------------------------- Step 14: load model
model, MODEL_NAME, DATASET_LABEL, RAW_FEATURES, MODEL_ERROR = None, "not loaded", "", [], None
try:
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    MODEL_NAME = bundle["model_name"]
    DATASET_LABEL = bundle.get("dataset_label", "")
    RAW_FEATURES = bundle["features"]
except FileNotFoundError:
    MODEL_ERROR = "The model file loan_model.joblib was not found. Run 'python train_model.py' first."
except Exception as exc:  # e.g. the model was saved with a very different library version
    MODEL_ERROR = (f"The saved model could not be loaded ({exc}). "
                   "Re-run 'python train_model.py' in this environment.")

database.init_db()
app = Flask(__name__)


# ------------------------------------------------------------ form options
def _options(*values):
    return [(v, v) for v in values]


OPTIONS = {
    "loan_purpose": [("Home", "Home purchase"), ("Vehicle", "Vehicle"), ("Education", "Education"),
                     ("Business", "Business"), ("Personal", "Personal"), ("Medical", "Medical")],
    "Gender": _options("Male", "Female"),
    "Married": _options("Yes", "No"),
    "Dependents": [("0", "0"), ("1", "1"), ("2", "2"), ("3+", "3 or more")],
    "Education": [("Graduate", "Graduate"), ("Not Graduate", "Not graduate")],
    "Self_Employed": _options("No", "Yes"),
    "Property_Area": [("Urban", "Urban"), ("Semiurban", "Semi-urban"), ("Rural", "Rural")],
    "Loan_Amount_Term": [(str(m), f"{m} months ({m / 12:g} yrs)")
                         for m in (12, 36, 60, 84, 120, 180, 240, 300, 360, 480)],
    "Credit_History": [("1", "Yes, has credit history"), ("0", "No credit history")],
}
CHOICE_FIELDS = list(OPTIONS.keys())

# field -> (label, min, max, type)
NUMBER_FIELDS = {
    "Age": ("Age", 18, 100, int),
    "ApplicantIncome": ("Applicant income (monthly)", 0, 10_000_000, float),
    "CoapplicantIncome": ("Co-applicant income (monthly)", 0, 10_000_000, float),
    "LoanAmount": ("Loan amount (in thousands)", 1, 100_000, float),
}

DEFAULTS = {
    "applicant_name": "",
    "Age": "30",
    "loan_purpose": "Home",
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "0",
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": "5000",
    "CoapplicantIncome": "0",
    "LoanAmount": "130",
    "Loan_Amount_Term": "360",
    "Property_Area": "Urban",
    "Credit_History": "1",
}


# ------------------------------------------------------- Step 15: input checks
def parse_form(form):
    """Validate the submitted form. Returns (clean_values, errors)."""
    values, errors = {}, []

    name = form.get("applicant_name", "").strip()
    if not name:
        errors.append("Applicant name is required.")
    elif len(name) > 80:
        errors.append("Applicant name must be 80 characters or fewer.")
    values["applicant_name"] = name

    for field in CHOICE_FIELDS:
        value = form.get(field, "").strip()
        allowed = [v for v, _ in OPTIONS[field]]
        if value not in allowed:
            errors.append(f"Please choose a valid option for {field.replace('_', ' ')}.")
        values[field] = value

    for field, (label, low, high, kind) in NUMBER_FIELDS.items():
        raw = form.get(field, "").strip()
        try:
            number = kind(float(raw))
        except ValueError:
            errors.append(f"{label} must be a number.")
            continue
        if not low <= number <= high:
            errors.append(f"{label} must be between {low:,} and {high:,}.")
        values[field] = number

    if errors:
        return values, errors

    # Convert coded fields to the types used in training
    values["Credit_History"] = float(values["Credit_History"])
    values["Loan_Amount_Term"] = float(values["Loan_Amount_Term"])
    return values, errors


# ----------------------------------------------------- Steps 16-17: predict
def assess(parsed):
    """Run the saved pipeline on one applicant and build the record to store."""
    frame = pd.DataFrame([{k: parsed[k] for k in RAW_FEATURES}])
    prob_approved = float(model.predict_proba(frame)[0][1])
    predicted = int(model.predict(frame)[0])
    derived = add_derived_features(frame).iloc[0]

    return {
        "reference": "LA-" + uuid.uuid4().hex[:8].upper(),
        "applicant_name": parsed["applicant_name"],
        "age": parsed["Age"],
        "loan_purpose": parsed["loan_purpose"],
        "loan_amount": parsed["LoanAmount"],
        "decision": "Approved" if predicted == 1 else "Not approved",
        "probability": round(prob_approved * 100, 1),
        "model_name": MODEL_NAME,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "inputs": parsed,
        "derived": {
            "TotalIncome": float(derived["TotalIncome"]),
            "EMI": float(derived["EMI"]),
            "DTI": float(derived["DTI"]),
        },
    }


def debt_band(dti):
    """Rule-of-thumb label for the debt-to-income ratio (a common guideline, not a bank rule)."""
    if dti is None or pd.isna(dti):
        return "Not available"
    if dti <= 0.36:
        return "Comfortable"
    if dti <= 0.50:
        return "Stretched"
    return "High"


# ------------------------------------------------------------ template helpers
@app.context_processor
def inject_globals():
    footer = f"Model: {MODEL_NAME}"
    if DATASET_LABEL:
        footer += f"  ·  {DATASET_LABEL}"
    footer += "  ·  Learning demo, not a real lending decision."
    return {"footer_note": footer, "model_error": MODEL_ERROR}


@app.template_filter("money")
def money(value):
    return f"{value:,.0f}"


@app.template_filter("pct")
def pct(value):
    return f"{value * 100:.1f}%"


# ------------------------------------------------------------------ routes
def _render_form(values, errors):
    return render_template("index.html", options=OPTIONS, values=values, errors=errors)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return _render_form(DEFAULTS, [])

    if model is None:
        return _render_form(request.form.to_dict(), [MODEL_ERROR])

    parsed, errors = parse_form(request.form)
    if errors:
        return _render_form(request.form.to_dict(), errors)

    record = assess(parsed)
    database.save_application(record)
    # Redirect so that refreshing the page shows the result without resubmitting the form
    return redirect(url_for("result", reference=record["reference"]))


@app.route("/result/<reference>")
def result(reference):
    record = database.get_application(reference)
    if record is None:
        abort(404)
    dti = record["derived"]["DTI"]
    return render_template("result.html", r=record, debt_band=debt_band(dti))


@app.route("/history")
def history():
    query = request.args.get("q", "").strip()
    decision = request.args.get("decision", "All")
    rows = database.search_applications(query, decision)
    return render_template("history.html", rows=rows, stats=database.get_stats(),
                           q=query, decision=decision)


@app.post("/history/<reference>/delete")
def delete(reference):
    database.delete_application(reference)
    return redirect(url_for("history"))


@app.errorhandler(404)
def not_found(_):
    return render_template("message.html", title="Not found",
                           message="That application reference does not exist."), 404


if __name__ == "__main__":
    print("Loan Approval Predictor: open http://127.0.0.1:5000 (press Ctrl+C to stop)")
    app.run(host="127.0.0.1", port=5000, debug=False)
