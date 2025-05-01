from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import itertools
import os
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load your trained XGBoost model
model = joblib.load("xgboost_traffic_model.pkl")
# Define possible green timings (30 to 40 with step of 2)
green_timings = list(range(30, 42, 2))
green_combinations = list(itertools.combinations(green_timings, 2))  # 36 combinations

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        # Input features: 8 lane volumes
        input_features = [
            data["lane1"], data["lane2"], data["lane3"], data["lane4"],
            data["lane5"], data["lane6"], data["lane7"], data["lane8"]
        ]
        # Create input set for all green timing combinations
        input_variants = [input_features + [g1, g2] for g1, g2 in green_combinations]
        input_df = pd.DataFrame(input_variants, columns=['X1','X2','X3','X4','X5','X6','X7','X8','green_1','green_2'])


        # Predict delays for each combination
        predicted_delays = model.predict(input_df)
        # Get the combination with minimum predicted delay
        min_index = np.argmin(predicted_delays)
        optimal_combination = green_combinations[min_index]
        optimal_delay = predicted_delays[min_index]

        green_1 = int(optimal_combination[0])
        green_2 = int(optimal_combination[1])
        min_delay = float(optimal_delay)

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
