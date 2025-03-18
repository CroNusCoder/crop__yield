from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Define the absolute path for the models folder
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_folder = os.path.join(project_root, "models")

# Load the model
model_path = os.path.join(models_folder, "crop_health_model.pkl")
try:
    model = joblib.load(model_path)
    print(f"Model loaded from: {model_path}")
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}")
    exit()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    input_data = pd.DataFrame([data])
    prediction = model.predict(input_data)
    return jsonify({"prediction": prediction[0]})

if __name__ == "__main__":
    app.run(debug=True)