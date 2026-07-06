import io
import logging
import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, send_file

# --- App / logging setup -----------------------------------------------
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fix 1: resolve pickle paths relative to this file, not the CWD, so the
# app works regardless of where it's launched from (local, Render, etc.)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "heart_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# Exact column names matching scaler.feature_names_in_
FEATURE_NAMES = [
    'Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol',
    'FBS over 120', 'EKG results', 'Max HR', 'Exercise angina',
    'ST depression', 'Slope of ST', 'Number of vessels fluro', 'Thallium'
]

# Fix 2: the "Presence/Absence" -> 0/1 mapping is NOT recorded anywhere
# inside the pickled model (no feature_names_in_/label encoder was saved
# with it), so it can't be verified programmatically. It's pulled out into
# a single named constant with an explicit warning so it can't be missed
# or silently drift if the model is ever retrained.
#
# !!! IMPORTANT !!!
# Verified against the training notebook:
#   df["Heart Disease"] = df["Heart Disease"].map({"Presence": 1, "Absence": 0})
#   model.classes_ == [0, 1]
# => 1 means "heart disease present", 0 means "absence".
# If heart_model.pkl is ever regenerated, re-verify this against
# model.classes_ and the label encoding used during training before
# trusting predictions.
DISEASE_CLASS_VALUE = 1  # class value that means "heart disease present"

# Fix 6: server-side validation so the model never sees out-of-range or
# nonsensical values, even if a request bypasses the HTML form entirely
# (e.g. curl/Postman), since the HTML min/max attributes are client-side only.
CATEGORICAL_ALLOWED = {
    'Sex': {0, 1},
    'Chest pain type': {1, 2, 3, 4},
    'FBS over 120': {0, 1},
    'EKG results': {0, 1, 2},
    'Exercise angina': {0, 1},
    'Slope of ST': {1, 2, 3},
    'Number of vessels fluro': {0, 1, 2, 3},
    'Thallium': {3, 6, 7},
}
CONTINUOUS_RANGES = {
    'Age': (1, 120),
    'BP': (1, 300),
    'Cholesterol': (1, 700),
    'Max HR': (1, 250),
    'ST depression': (0.0, 10.0),
}


def validate_feature(name, value):
    """Raise ValueError with a clear message if value is out of range."""
    if name in CATEGORICAL_ALLOWED:
        if int(value) not in CATEGORICAL_ALLOWED[name]:
            allowed = sorted(CATEGORICAL_ALLOWED[name])
            raise ValueError(f"'{name}' must be one of {allowed}, got {value}")
    elif name in CONTINUOUS_RANGES:
        low, high = CONTINUOUS_RANGES[name]
        if not (low <= value <= high):
            raise ValueError(f"'{name}' must be between {low} and {high}, got {value}")


def run_model_on_row(input_values):
    """Shared prediction logic for a single row of already-validated values.

    Returns True if the model predicts heart disease present, else False.
    """
    features_df = pd.DataFrame([input_values], columns=FEATURE_NAMES)

    # If the scaler remembers its fit column order, force-align to it
    if hasattr(scaler, "feature_names_in_"):
        features_df = features_df[list(scaler.feature_names_in_)]

    features_scaled = scaler.transform(features_df)
    prediction = model.predict(features_scaled)[0]

    # Normalize prediction to a plain Python value for reliable comparison
    pred_value = prediction.item() if isinstance(prediction, np.generic) else prediction

    # Handle both numeric-encoded (0/1) and string-encoded ('Presence'/'Absence') labels
    if isinstance(pred_value, str):
        is_disease = pred_value.strip().lower() == 'presence'
    else:
        is_disease = int(pred_value) == DISEASE_CLASS_VALUE

    return is_disease


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    # Fix 7: input/validation errors (bad or missing form data) are handled
    # separately from unexpected server errors, so the two don't look
    # identical to the end user and unexpected failures aren't leaked
    # verbatim into the rendered page.
    try:
        input_values = []
        for feature_name in FEATURE_NAMES:
            if feature_name not in request.form or request.form[feature_name] == "":
                raise ValueError(f"Missing value for '{feature_name}'")
            value = float(request.form[feature_name])
            validate_feature(feature_name, value)
            input_values.append(value)

        is_disease = run_model_on_row(input_values)

        if is_disease:
            result = "Heart Disease Detected"
            risk = "high"
        else:
            result = "No Heart Disease Detected"
            risk = "low"

    except (ValueError, KeyError) as e:
        # Bad/missing/out-of-range input submitted by the client.
        logger.warning("Invalid input on /predict: %s", e)
        result = f"Invalid input: {e}"
        risk = "error"

    except Exception:
        # Anything else is an unexpected server-side failure. Log the full
        # detail server-side but don't expose internals to the user.
        logger.exception("Unexpected error during prediction")
        result = "Something went wrong while processing your request. Please try again."
        risk = "error"

    return render_template("index.html", prediction_text=result, risk=risk)


@app.route('/predict_csv', methods=['GET', 'POST'])
def predict_csv():
    """Bulk prediction: upload a CSV of many patients, get back a CSV with
    a Prediction column added for every row, as a direct file download.
    Rows with missing/invalid values are skipped and flagged rather than
    crashing the whole batch.
    """
    if request.method == 'GET':
        return render_template('upload_csv.html')

    file = request.files.get('file')
    if not file or file.filename == '':
        return render_template('upload_csv.html', error="Please choose a CSV file to upload.")

    if not file.filename.lower().endswith('.csv'):
        return render_template('upload_csv.html', error="Only .csv files are supported.")

    try:
        df = pd.read_csv(file)
    except Exception as e:
        logger.warning("Could not parse uploaded CSV: %s", e)
        return render_template('upload_csv.html', error=f"Could not read this file as CSV: {e}")

    missing_cols = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing_cols:
        return render_template(
            'upload_csv.html',
            error=f"Your CSV is missing required column(s): {', '.join(missing_cols)}. "
                  f"Expected columns: {', '.join(FEATURE_NAMES)}"
        )

    results = []
    for idx, row in df.iterrows():
        row_errors = []
        input_values = []
        for feature_name in FEATURE_NAMES:
            raw_value = row[feature_name]
            try:
                if pd.isna(raw_value):
                    raise ValueError(f"Missing value for '{feature_name}'")
                value = float(raw_value)
                validate_feature(feature_name, value)
                input_values.append(value)
            except (ValueError, TypeError) as e:
                row_errors.append(str(e))

        if row_errors:
            results.append({"Prediction": "Skipped - invalid row", "Details": "; ".join(row_errors)})
            continue

        try:
            is_disease = run_model_on_row(input_values)
            results.append({
                "Prediction": "Heart Disease Detected" if is_disease else "No Heart Disease Detected",
                "Details": ""
            })
        except Exception:
            logger.exception("Unexpected error predicting row %s", idx)
            results.append({"Prediction": "Error - could not predict", "Details": "Server error, check logs"})

    results_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)

    buffer = io.StringIO()
    results_df.to_csv(buffer, index=False)
    byte_buffer = io.BytesIO(buffer.getvalue().encode('utf-8'))

    return send_file(
        byte_buffer,
        mimetype='text/csv',
        as_attachment=True,
        download_name='heart_disease_predictions.csv'
    )


if __name__ == "__main__":
    # Fix 4: debug mode must never run in production (Flask's debugger
    # allows remote code execution if the endpoint is reachable). It now
    # defaults to off and is only enabled via an explicit env var for
    # local development.
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug_mode)
