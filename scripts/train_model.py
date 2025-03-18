import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Define the absolute path for the models folder
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_folder = os.path.join(project_root, "models")

# Create the models folder if it doesn't exist
if not os.path.exists(models_folder):
    os.makedirs(models_folder)

# Load combined data
data = pd.read_csv("../data/combined_data.csv")

# Define features (X) and target (y)
X = data[["temperature", "humidity", "rainfall", "soil_fertility_index"]]
y = data["crop_yield"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Save the model
model_path = os.path.join(models_folder, "crop_health_model.pkl")
joblib.dump(model, model_path)
print(f"Model saved to: {model_path}")