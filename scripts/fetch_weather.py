import requests
import pandas as pd

API_KEY = "api_key"

def fetch_current_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return {
            "city": city,
            "temperature": data["main"]["temp"] - 273.15,  # Convert Kelvin to Celsius
            "humidity": data["main"]["humidity"],
            "rainfall": data.get("rain", {}).get("1h", 0),  # Rainfall in the last hour
        }
    else:
        print("Error fetching weather data")
        return None


cities = ["New Delhi", "Mumbai", "Bangalore"]
weather_data = [fetch_current_weather(city) for city in cities]

# csv saver
df = pd.DataFrame(weather_data)
df.to_csv("../data/weather_data.csv", index=False)
print("Weather data saved to data/weather_data.csv")