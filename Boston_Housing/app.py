from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

model = pickle.load(open("boston_model.pkl", "rb"))
scaler = pickle.load(open("boston_scaler.pkl", "rb"))

# IMPORTANT: feature names must match training data order
FEATURE_NAMES = [
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
    "LSTAT"
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = []

        for feature in FEATURE_NAMES:
            value = float(request.form[feature])
            input_data.append(value)

        input_array = np.array(input_data).reshape(1, -1)

        input_scaled = scaler.transform(input_array)

        prediction = model.predict(input_scaled)[0]

        prediction = round(prediction, 2)

        return render_template(
            "index.html",
            prediction_text=f"Predicted House Price: {prediction}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {e}"
        )

if __name__ == "__main__":
    app.run(debug=True)
