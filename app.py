from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load your trained XGBoost model
model = joblib.load("xgboost_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Input features: 8 lane volumes
        input_features = [
            data["lane1"], data["lane2"], data["lane3"], data["lane4"],
            data["lane5"], data["lane6"], data["lane7"], data["lane8"]
        ]
        input_array = np.array([input_features])

        # Predict returns [green_1, green_2, min_delay]
        result = model.predict(input_array)[0]  

        green_1 = int(result[0])
        green_2 = int(result[1])
        min_delay = float(result[2])

        return jsonify({
            "green_1": green_1,
            "green_2": green_2,
            "predicted_minimum_delay": min_delay
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/", methods=["GET"])
def home():
    return "XGBoost Optimal Green Time API is running!"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
