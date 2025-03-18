import pandas as pd
import os

# Define the absolute path for the data folder
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_folder = os.path.join(project_root, "data")

# Print the absolute path of the data folder
print(f"Data folder: {data_folder}")

# Load datasets
try:
    weather_data = pd.read_csv(os.path.join(data_folder, "weather_data.csv"))
    soil_data = pd.read_csv(os.path.join(data_folder, "soil_data.csv"))
    crop_yield_data = pd.read_csv(os.path.join(data_folder, "crop_yield.csv"))
    print("Input files loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading input files: {e}")
    exit()

# Print the first few rows of each dataset
print("\nWeather Data:")
print(weather_data.head())

print("\nSoil Data:")
print(soil_data.head())

print("\nCrop Yield Data:")
print(crop_yield_data.head())

# Merge datasets on city
try:
    combined_data = pd.merge(weather_data, soil_data, on="city")
    combined_data = pd.merge(combined_data, crop_yield_data, on="city")
    print("\nDatasets merged successfully.")
except KeyError as e:
    print(f"Error merging datasets: {e}")
    exit()

# Feature engineering
combined_data["soil_fertility_index"] = combined_data["nitrogen"] + combined_data["phosphorus"] + combined_data["potassium"]

# Save combined data
output_path = os.path.join(data_folder, "combined_data.csv")
combined_data.to_csv(output_path, index=False)
print(f"Combined data saved to: {output_path}")