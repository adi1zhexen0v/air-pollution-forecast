from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

@app.route("/api/predictions")
def get_predictions():
    prediction_dir = os.path.join("..", "outputs", "predictions")
    files = [f for f in os.listdir(prediction_dir) if f.startswith("predicted_pm25") and f.endswith(".csv")]

    if not files:
        return jsonify({"error": "No predictions found"}), 404

    latest_file = max(files, key=lambda f: os.path.getctime(os.path.join(prediction_dir, f)))
    df = pd.read_csv(os.path.join(prediction_dir, latest_file))

    return jsonify(df.to_dict(orient="records"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

