import pandas as pd
import numpy as np

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# 1. LOAD DATA
# ============================================================

train = pd.read_csv("train_data.csv")
test = pd.read_csv("test_data.csv")

train["DateTime"] = pd.to_datetime(train["DateTime"])
test["DateTime"] = pd.to_datetime(test["DateTime"])

train = train.sort_values("DateTime")
test = test.sort_values("DateTime")

print("Training data:", train.shape)
print("Testing data :", test.shape)


# ============================================================
# 2. CREATE 15-MINUTE TIME SERIES
# ============================================================

train_series = (
    train.set_index("DateTime")["Sales Count"]
    .resample("15min")
    .sum()
    .fillna(0)
)

test_series = (
    test.set_index("DateTime")["Sales Count"]
    .resample("15min")
    .sum()
    .fillna(0)
)

print("\n15-minute training observations:", len(train_series))
print("15-minute testing observations :", len(test_series))


# ============================================================
# 3. USE RECENT TRAINING DATA
# ============================================================

# Use the most recent 2,000 observations
# to make ARIMA computationally manageable.

MAX_TRAIN_POINTS = 2000

if len(train_series) > MAX_TRAIN_POINTS:
    arima_train = train_series.iloc[-MAX_TRAIN_POINTS:]
else:
    arima_train = train_series

print("\nARIMA training observations:", len(arima_train))


# ============================================================
# 4. ARIMA MODEL
# ============================================================

print("\n==============================")
print("ARIMA MODEL")
print("==============================")

print("\nTraining ARIMA model...")

model = ARIMA(
    arima_train,
    order=(1, 1, 1)
)

model_fit = model.fit()

print("ARIMA training completed.")


# ============================================================
# 5. FORECAST
# ============================================================

print("\nGenerating forecasts...")

forecast = model_fit.forecast(
    steps=len(test_series)
)

forecast = np.maximum(
    forecast,
    0
)

forecast.index = test_series.index

# Save ARIMA forecast to a separate CSV file
forecast_df = pd.DataFrame({
    "DateTime": test_series.index,
    "Forecast": forecast
})

forecast_df.to_csv(
    "arima_forecast.csv",
    index=False
)

print("ARIMA forecast saved as: arima_forecast.csv")

# ============================================================
# 6. EVALUATION FUNCTION
# ============================================================

def calculate_metrics(actual, predicted):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    actual = np.array(actual)
    predicted = np.array(predicted)

    non_zero = actual != 0

    if non_zero.sum() > 0:

        mape = np.mean(
            np.abs(
                (actual[non_zero] - predicted[non_zero])
                / actual[non_zero]
            )
        ) * 100

    else:
        mape = np.nan

    return mae, rmse, mape


# ============================================================
# 7. FORECAST HORIZONS
# ============================================================

print("\n==============================")
print("ARIMA FORECAST RESULTS")
print("==============================")

horizons = {
    "15 Minutes": 1,
    "30 Minutes": 2,
    "1 Hour": 4,
    "2 Hours": 8
}

results = []

for horizon, steps in horizons.items():

    actual = test_series.iloc[:steps]
    predicted = forecast.iloc[:steps]

    mae, rmse, mape = calculate_metrics(
        actual,
        predicted
    )

    results.append({
        "Model": "ARIMA",
        "Horizon": horizon,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape
    })

    print(f"\n{horizon}")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"MAPE : {mape:.2f}%")


# ============================================================
# 8. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

print("\n==============================")
print("ARIMA RESULTS")
print("==============================")

print(results_df.to_string(index=False))

results_df.to_csv(
    "arima_results.csv",
    index=False
)

print("\nResults saved as:")
print("arima_results.csv")