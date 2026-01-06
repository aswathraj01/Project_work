from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("wine_type_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        fixed_acidity = float(request.form["fixed_acidity"])
        volatile_acidity = float(request.form["volatile_acidity"])
        citric_acid = float(request.form["citric_acid"])
        residual_sugar = float(request.form["residual_sugar"])
        chlorides = float(request.form["chlorides"])
        free_sulfur_dioxide = float(request.form["free_sulfur_dioxide"])
        total_sulfur_dioxide = float(request.form["total_sulfur_dioxide"])
        density = float(request.form["density"])
        pH = float(request.form["pH"])
        sulphates = float(request.form["sulphates"])
        alcohol = float(request.form["alcohol"])

        features = np.array([[
            fixed_acidity,
            volatile_acidity,
            citric_acid,
            residual_sugar,
            chlorides,
            free_sulfur_dioxide,
            total_sulfur_dioxide,
            density,
            pH,
            sulphates,
            alcohol
        ]])

        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]

        wine_type = "White" if prediction == 1 else "Red"

        return render_template(
            "index.html",
            prediction_text=f"Predicted Wine Type: {wine_type}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )

if __name__ == "__main__":
    app.run(debug=True)
