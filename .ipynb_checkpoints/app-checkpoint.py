from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load saved model & scaler
model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    # Pass None for prediction initially
    return render_template("index.html", prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data safely
        fields = ['gender', 'age', 'smoking', 'yellow_fingers', 'anxiety',
                  'peer_pressure', 'chronic_disease', 'fatigue', 'allergy',
                  'wheezing', 'alcohol', 'coughing', 'shortness_of_breath',
                  'swallowing_difficulty', 'chest_pain']

        # Convert all to int
        features = []
        for f in fields:
            value = request.form.get(f)
            if value is None or value == '':
                return f"Error: Missing input for {f}", 400
            features.append(int(value))

        features = np.array([features])

        # Scale features
        features = scaler.transform(features)

        # Predict
        prediction = model.predict(features)[0]
        result = "LUNG CANCER DETECTED" if prediction == 1 else "NO LUNG CANCER"

        # Render result in index.html
        return render_template("index.html", prediction=result)

    except Exception as e:
        return f"Error: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True)
