"""
Open-Meteo helpers for the final London Bikes model.

Both functions return daily values for the exact numeric predictors used by the
final exported model:

    tempmax, windspeed, solarradiation, sealevelpressure, visibility, precip

The API is requested hourly and aggregated to daily values so units align with
the London Bikes training variables as closely as possible:

- tempmax: maximum hourly temperature_2m, °C
- windspeed: mean hourly wind_speed_10m, km/h
- solarradiation: mean hourly shortwave_radiation, W/m²
- sealevelpressure: mean hourly pressure_msl, hPa
- visibility: mean hourly visibility converted from metres to km
- precip: sum of hourly precipitation, mm
"""

import requests
import pandas as pd

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

HOURLY_FIELDS = {
    "temperature_2m": "temperature_2m",
    "wind_speed_10m": "windspeed",
    "shortwave_radiation": "solarradiation",
    "pressure_msl": "sealevelpressure",
    "visibility": "visibility_m",
    "precipitation": "precip",
}


def geocode(location):
    resp = requests.get(
        GEOCODE_URL,
        params={"name": location, "count": 1, "language": "en"},
        timeout=15,
    )
    resp.raise_for_status()
    results = resp.json().get("results")
    if not results:
        raise ValueError(f"Open-Meteo could not find a location called {location!r}.")
    top = results[0]
    label = ", ".join(p for p in [top.get("name"), top.get("country")] if p)
    return top["latitude"], top["longitude"], label


def _hourly_to_daily(hourly, label):
    hf = pd.DataFrame(
        {
            output_name: hourly[api_name]
            for api_name, output_name in HOURLY_FIELDS.items()
        }
    )
    hf["time"] = pd.to_datetime(hourly["time"])
    hf["date"] = hf["time"].dt.normalize()

    for col in HOURLY_FIELDS.values():
        hf[col] = pd.to_numeric(hf[col], errors="coerce")

    daily = hf.groupby("date").agg(
        tempmax=("temperature_2m", "max"),
        windspeed=("windspeed", "mean"),
        solarradiation=("solarradiation", "mean"),
        sealevelpressure=("sealevelpressure", "mean"),
        visibility_m=("visibility_m", "mean"),
        precip=("precip", "sum"),
    ).reset_index()

    daily["visibility"] = daily["visibility_m"] / 1000.0
    daily = daily.drop(columns=["visibility_m"])
    daily.insert(1, "day_of_week", daily["date"].dt.strftime("%a"))

    daily.attrs["location"] = label
    return daily


def _request_hourly(url, params, timeout):
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    if "hourly" not in payload:
        raise ValueError("Open-Meteo returned no hourly weather data.")
    return payload["hourly"]


def open_meteo(location="London", days_to_forecast=5):
    """Return the next 1–7 days of weather for the final demand model."""
    days_to_forecast = int(days_to_forecast)
    if not 1 <= days_to_forecast <= 7:
        raise ValueError("days_to_forecast must be between 1 and 7.")

    lat, lon, label = geocode(location)

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(HOURLY_FIELDS.keys()),
        "forecast_days": days_to_forecast,
        "timezone": "auto",
        "wind_speed_unit": "kmh",
    }

    hourly = _request_hourly(FORECAST_URL, params, timeout=20)
    return _hourly_to_daily(hourly, label)


def open_meteo_history(location, start_date, end_date):
    """Return historical daily weather for the final demand model."""
    lat, lon, label = geocode(location)

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_FIELDS.keys()),
        "timezone": "auto",
        "wind_speed_unit": "kmh",
    }

    hourly = _request_hourly(ARCHIVE_URL, params, timeout=30)
    return _hourly_to_daily(hourly, label)


if __name__ == "__main__":
    print(open_meteo("London", 5).to_string(index=False))
    print(open_meteo_history("London", "2026-01-01", "2026-01-07").to_string(index=False))
