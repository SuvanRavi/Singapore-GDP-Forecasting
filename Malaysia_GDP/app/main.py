from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
import os

app = FastAPI(
    title="Singapore GDP Forecast API",
    description=(
        "==> This API uses a Prophet-based time series model to forecast Singapore's quarterly GDP in unit of **SGD Million**.\n\n"
        "==> Use `/predict` to retrieve GDP forecasts for a specified year **between 2025 and 2030**.\n\n"
        "==> The model is trained with macroeconomic indicators and returns predicted GDP values along with confidence intervals.\n\n"
        "==> User only has to input a year between 2025 and 2030\n"
    ),
    version="1.0.0"
)

MODEL_PATH = os.path.join("app", "sg_gdp_prophet.pkl")
CSV_PATH = os.path.join("app", "FutureValues.csv")

# Load model and future values
model = joblib.load(MODEL_PATH)
df_future = pd.read_csv(CSV_PATH, parse_dates=["ds"])

@app.get("/predict")
def predict_gdp(
    year: int = Query(..., ge=2025, le=2030)
):
    # Filter future values for the requested year
    df_year = df_future[df_future["ds"].dt.year == year]
    if df_year.empty:
        return JSONResponse({"error": "No data available for the specified year."}, status_code=404)

    # Make forecast
    forecast = model.predict(df_year)
    forecast["quarter"] = forecast["ds"].dt.quarter.apply(lambda q: f"Q{q}")
    forecast = forecast.round(2)

    # Construct response
    # 'ds' and 'yhat' are the date and predicted GDP value respectively
    result = []
    for _, row in forecast.iterrows():
        result.append({
            "quarter": row["quarter"],
            "gdp_prediction": row["yhat"],
            "upper_boundary": row["yhat_upper"],
            "lower_boundary": row["yhat_lower"]
        })

    return {
        "year": year,
        "model": "Prophet GDP Forecaster",
        "GDP Value": result
    }
