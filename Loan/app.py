from flask import Flask, render_template, request
import pickle
import numpy as np


app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form values
        gender = int(request.form["Gender"])
        married = int(request.form["Married"])
        dependents = int(request.form["Dependents"])
        education = int(request.form["Education"])
        self_employed = int(request.form["Self_Employed"])
        applicant_income = float(request.form["ApplicantIncome"])
        coapplicant_income = float(request.form["CoapplicantIncome"])
        loan_amount = float(request.form["LoanAmount"])
        loan_term = float(request.form["Loan_Amount_Term"])
        credit_history = int(request.form["Credit_History"])
        prop_semiurban = int(request.form["Property_Area_Semiurban"])
        prop_urban = int(request.form["Property_Area_Urban"])

        # Arrange features in SAME ORDER as training
        features = np.array([[  
            gender,
            married,
            dependents,
            education,
            self_employed,
            applicant_income,
            coapplicant_income,
            loan_amount,
            loan_term,
            credit_history,
            prop_semiurban,
            prop_urban
        ]])

        # Scale numerical values (same scaler used in training)
        features[:, 5:8] = scaler.transform(features[:, 5:8])

        prediction = model.predict(features)[0]

        result = "Loan Approved ✅" if prediction == 1 else "Loan Rejected ❌"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}")
    
if __name__ == "__main__":
    app.run(debug=True)