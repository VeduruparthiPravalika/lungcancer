from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load saved model & scaler
model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        gender = int(request.form['gender'])
        age = int(request.form['age'])
        smoking = int(request.form['smoking'])
        yellow_fingers = int(request.form['yellow_fingers'])
        anxiety = int(request.form['anxiety'])
        peer_pressure = int(request.form['peer_pressure'])
        chronic_disease = int(request.form['chronic_disease'])
        fatigue = int(request.form['fatigue'])
        allergy = int(request.form['allergy'])
        wheezing = int(request.form['wheezing'])
        alcohol = int(request.form['alcohol'])
        coughing = int(request.form['coughing'])
        breath_shortness = int(request.form['breath_shortness'])
        swallowing_difficulty = int(request.form['swallowing_difficulty'])
        chest_pain = int(request.form['chest_pain'])

        features = np.array([[gender, age, smoking, yellow_fingers, anxiety,
                              peer_pressure, chronic_disease, fatigue, allergy,
                              wheezing, alcohol, coughing, breath_shortness,
                              swallowing_difficulty, chest_pain]])

        # Scale features
        features = scaler.transform(features)

        prediction = model.predict(features)[0]

        result = "LUNG CANCER DETECTED" if prediction == 1 else "NO LUNG CANCER"

        return render_template("result.html", prediction=result)

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)
