import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA


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
# 3. USE SAME ARIMA TRAINING SETUP
# ============================================================

MAX_TRAIN_POINTS = 2000

if len(train_series) > MAX_TRAIN_POINTS:
    arima_train = train_series.iloc[-MAX_TRAIN_POINTS:]
else:
    arima_train = train_series

print("\nARIMA training observations:", len(arima_train))


# ============================================================
# 4. TRAIN ARIMA
# ============================================================

print("\n==============================")
print("ARIMA UNCERTAINTY ANALYSIS")
print("==============================")

print("\nTraining ARIMA model...")

model = ARIMA(
    arima_train,
    order=(1, 1, 1)
)

model_fit = model.fit()

print("ARIMA training completed.")


# ============================================================
# 5. GENERATE FORECAST + CONFIDENCE INTERVAL
# ============================================================

print("\nGenerating forecast and prediction intervals...")

forecast_result = model_fit.get_forecast(
    steps=8
)

forecast = forecast_result.predicted_mean

confidence = forecast_result.conf_int(
    alpha=0.05
)

# Prevent negative ticket predictions
forecast = np.maximum(forecast, 0)

confidence.iloc[:, 0] = np.maximum(
    confidence.iloc[:, 0],
    0
)

confidence.iloc[:, 1] = np.maximum(
    confidence.iloc[:, 1],
    0
)

forecast.index = test_series.index[:8]
confidence.index = test_series.index[:8]


# ============================================================
# 6. CREATE FORECAST TABLE
# ============================================================

forecast_table = pd.DataFrame({
    "DateTime": forecast.index,
    "Actual Sales": test_series.iloc[:8].values,
    "Forecast": forecast.values,
    "Lower Bound": confidence.iloc[:, 0].values,
    "Upper Bound": confidence.iloc[:, 1].values
})

forecast_table["Interval Width"] = (
    forecast_table["Upper Bound"]
    - forecast_table["Lower Bound"]
)


print("\n==============================")
print("FORECAST WITH 95% PREDICTION INTERVAL")
print("==============================")

print(
    forecast_table.to_string(index=False)
)


# ============================================================
# 7. CONFIDENCE BAND SUMMARY
# ============================================================

print("\n==============================")
print("UNCERTAINTY SUMMARY")
print("==============================")

print(
    f"Average interval width: "
    f"{forecast_table['Interval Width'].mean():.2f}"
)

print(
    f"Minimum interval width: "
    f"{forecast_table['Interval Width'].min():.2f}"
)

print(
    f"Maximum interval width: "
    f"{forecast_table['Interval Width'].max():.2f}"
)


# ============================================================
# 8. SAVE FORECAST TABLE
# ============================================================

forecast_table.to_csv(
    "forecast_uncertainty.csv",
    index=False
)

print("\nFile saved:")
print("forecast_uncertainty.csv")


# ============================================================
# 9. VISUALIZE FORECAST AND CONFIDENCE BAND
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    forecast_table["DateTime"],
    forecast_table["Actual Sales"],
    marker="o",
    label="Actual Sales"
)

plt.plot(
    forecast_table["DateTime"],
    forecast_table["Forecast"],
    marker="o",
    label="ARIMA Forecast"
)

plt.fill_between(
    forecast_table["DateTime"],
    forecast_table["Lower Bound"],
    forecast_table["Upper Bound"],
    alpha=0.2,
    label="95% Prediction Interval"
)

plt.xlabel("Date and Time")
plt.ylabel("Ticket Sales")

plt.title(
    "ARIMA Ferry Ticket Demand Forecast with 95% Prediction Interval"
)

plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

plt.savefig(
    "arima_forecast_uncertainty.png",
    dpi=300
)

plt.show()

print("\nChart saved:")
print("arima_forecast_uncertainty.png")