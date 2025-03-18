import pandas as pd
import numpy as np
import os

# Define the absolute path for the data folder
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_folder = os.path.join(project_root, "data")

# Print the absolute path of the data folder
print(f"Data will be saved to: {data_folder}")

# Create the data folder if it doesn't exist
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

# Generate synthetic weather data
np.random.seed(42)
cities = ["City_A", "City_B", "City_C"]
weather_data = {
    "city": cities * 10,  # 10 years of data for each city
    "temperature": np.random.uniform(20, 35, 30),  # Random temperature between 20°C and 35°C
    "humidity": np.random.uniform(40, 80, 30),     # Random humidity between 40% and 80%
    "rainfall": np.random.uniform(0, 100, 30),     # Random rainfall between 0mm and 100mm
}
weather_df = pd.DataFrame(weather_data)

# Generate synthetic soil data
soil_data = {
    "city": cities,
    "nitrogen": np.random.uniform(100, 200, 3),  # Random nitrogen levels
    "phosphorus": np.random.uniform(50, 150, 3), # Random phosphorus levels
    "potassium": np.random.uniform(150, 300, 3), # Random potassium levels
}
soil_df = pd.DataFrame(soil_data)

# Generate synthetic crop yield data
crop_yield_data = {
    "city": cities * 10,  # 10 years of data for each city
    "crop_yield": np.random.uniform(1000, 5000, 30),  # Random crop yield between 1000 and 5000 kg/ha
}
crop_yield_df = pd.DataFrame(crop_yield_data)

# Save synthetic data to CSV files
weather_df.to_csv(os.path.join(data_folder, "weather_data.csv"), index=False)
soil_df.to_csv(os.path.join(data_folder, "soil_data.csv"), index=False)
crop_yield_df.to_csv(os.path.join(data_folder, "crop_yield.csv"), index=False)

print("Synthetic data generated and saved to data/ folder.")