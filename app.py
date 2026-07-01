from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request


app = Flask(__name__)

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "artifacts" / "model.joblib"

FEATURE_ORDER = [
    "CRIM",
    "ZN",
    "INDUS",
    "CHAS",
    "NOX",
    "RM",
    "AGE",
    "DIS",
    "RAD",
    "TAX",
    "PTRATIO",
    "B",
    "LSTAT",
]

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. "
        "Run 'dvc pull' before starting the application."
    )

model = joblib.load(MODEL_PATH)


def prepare_input(data: dict) -> pd.DataFrame:
    missing_features = [feature for feature in FEATURE_ORDER if feature not in data]

    if missing_features:
        raise ValueError(
            f"Missing required features: {', '.join(missing_features)}"
        )

    try:
        values = {feature: float(data[feature]) for feature in FEATURE_ORDER}
    except (TypeError, ValueError) as error:
        raise ValueError("All feature values must be numeric.") from error

    return pd.DataFrame([values], columns=FEATURE_ORDER)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "healthy",
            "model": MODEL_PATH.name,
        }
    )


@app.route("/predict_api", methods=["POST"])
def predict_api():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify(
            {
                "error": "Send a JSON object with all 13 required housing features."
            }
        ), 400

    try:
        input_df = prepare_input(data)
        prediction = float(model.predict(input_df)[0])

        return jsonify({"prediction": prediction})

    except ValueError as error:
        return jsonify({"error": str(error)}), 400


@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_df = prepare_input(request.form.to_dict())
        prediction = float(model.predict(input_df)[0])

        return render_template(
            "home.html",
            prediction_text=f"Predicted Price of the House is {prediction:.2f}",
        )

    except ValueError as error:
        return render_template(
            "home.html",
            prediction_text=f"Input error: {error}",
        ), 400


if __name__ == "__main__":
    app.run(debug=True)