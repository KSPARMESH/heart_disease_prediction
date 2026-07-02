from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
model = pickle.load(open("heart_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Exact column names matching scaler.feature_names_in_
FEATURE_NAMES = [
    'Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol',
    'FBS over 120', 'EKG results', 'Max HR', 'Exercise angina',
    'ST depression', 'Slope of ST', 'Number of vessels fluro', 'Thallium'
]

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Build input in the EXACT order the scaler/model were fit on
        input_values = [float(request.form[f]) for f in FEATURE_NAMES]
        features_df = pd.DataFrame([input_values], columns=FEATURE_NAMES)

        # If the scaler remembers its fit column order, force-align to it
        if hasattr(scaler, "feature_names_in_"):
            features_df = features_df[list(scaler.feature_names_in_)]

        features_scaled = scaler.transform(features_df)
        prediction = model.predict(features_scaled)[0]

        print("Raw prediction value:", prediction, type(prediction))

        # Normalize prediction to a plain Python value for reliable comparison
        if isinstance(prediction, (np.generic,)):
            pred_value = prediction.item()
        else:
            pred_value = prediction

        # Handle both numeric-encoded (0/1) and string-encoded ('Presence'/'Absence') labels
        if isinstance(pred_value, str):
            is_disease = pred_value.strip().lower() == 'presence'
        else:
            # Model encodes: 0 = Presence (Disease), 1 = Absence (No Disease)
            is_disease = int(pred_value) == 0

        if is_disease:
            result = "Heart Disease Detected"
            risk = "high"
        else:
            result = "No Heart Disease Detected"
            risk = "low"

    except Exception as e:
        print("Prediction error:", e)
        result = f"Error: {str(e)}"
        risk = "error"

    return render_template("index.html", prediction_text=result, risk=risk)

if __name__ == "__main__":
    app.run(debug=True)