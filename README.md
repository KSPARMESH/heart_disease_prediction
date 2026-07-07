# Heart Disease Prediction

A Flask-based web application that predicts the likelihood of heart disease in a patient using machine learning, trained on the UCI Heart Disease (Kaggle `heart.csv`) dataset.

## Overview

This project takes in a set of clinical parameters (age, sex, chest pain type, blood pressure, cholesterol, etc.) and uses a trained machine learning model to predict whether a patient is at risk of heart disease. The model is served through a simple Flask web interface where users can input patient data and get an instant prediction.

## Features

- Web form to input patient medical data
- Machine learning model trained on the UCI Heart Disease dataset
- Instant prediction of heart disease risk (Yes/No)
- Simple, lightweight Flask backend

## Dataset

The model is trained on the **UCI Heart Disease dataset** (commonly distributed as `heart.csv` on Kaggle), which includes attributes such as:

| Feature | Description |
|---|---|
| age | Age of the patient |
| sex | Sex (1 = male, 0 = female) |
| cp | Chest pain type |
| trestbps | Resting blood pressure |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl |
| restecg | Resting electrocardiographic results |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina |
| oldpeak | ST depression induced by exercise |
| slope | Slope of the peak exercise ST segment |
| ca | Number of major vessels colored by fluoroscopy |
| thal | Thalassemia |
| target | Diagnosis of heart disease (1 = present, 0 = absent) |

## Tech Stack

- **Backend:** Python, Flask
- **Machine Learning:** scikit-learn, pandas, numpy
- **Frontend:** HTML, CSS (Jinja2 templates)

## Project Structure

```
heart_disease_prediction/
├── app.py                 # Flask application entry point
├── model/                 # Trained ML model files (e.g., .pkl)
├── static/                # CSS/JS assets
├── templates/              # HTML templates
├── heart.csv               # Dataset
├── requirements.txt         # Python dependencies
└── README.md
```

> Note: Update this section to match your actual folder/file names if they differ.

## Installation

1. Clone the repository
   ```bash
   git clone https://github.com/KSPARMESH/heart_disease_prediction.git
   cd heart_disease_prediction
   ```

2. Create a virtual environment (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Flask app
   ```bash
   python app.py
   ```

2. Open your browser and go to:
   ```
   https://heart-disease-prediction-byfqbj6zz-parmesh.vercel.app/
   ```

3. Enter the patient's medical details in the form and click **Predict** to see the result.

## Model Training

If you'd like to retrain the model:

1. Ensure `heart.csv` is present in the project directory.
2. Run the training script (e.g., `train_model.py`) to preprocess the data, train the model, and save it as a `.pkl` file.
3. The Flask app loads this saved model to make predictions.

## Results

The model's performance can be evaluated using standard classification metrics such as accuracy, precision, recall, and F1-score. Update this section with your actual results, e.g.:

- Logistic Regression Accuracy: 88%
- Random Forest Accuracy: 79%

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

## Acknowledgements

- Dataset: [UCI Machine Learning Repository - Heart Disease Dataset](https://www.kaggle.com/code/farzadnekouei/heart-disease-prediction)
- Built with Flask and scikit-learn
