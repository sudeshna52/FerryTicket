import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# 1. LOAD TRAINING AND TEST DATA
# ============================================================

train = pd.read_csv("train_data.csv")
test = pd.read_csv("test_data.csv")

train["DateTime"] = pd.to_datetime(train["DateTime"])
test["DateTime"] = pd.to_datetime(test["DateTime"])

print("Training data:", train.shape)
print("Testing data:", test.shape)


# ============================================================
# 2. DEFINE FORECAST HORIZONS
# ============================================================

horizons = {
    "15 Minutes": "Target_15min",
    "30 Minutes": "Target_30min",
    "1 Hour": "Target_1hour",
    "2 Hours": "Target_2hour"
}


# ============================================================
# 3. EVALUATION FUNCTION
# ============================================================

def calculate_metrics(actual, predicted):

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    # Avoid division by zero
    actual_array = np.array(actual)
    predicted_array = np.array(predicted)

    non_zero = actual_array != 0

    if non_zero.sum() > 0:
        mape = np.mean(
            np.abs(
                (actual_array[non_zero] - predicted_array[non_zero])
                / actual_array[non_zero]
            )
        ) * 100
    else:
        mape = np.nan

    return mae, rmse, mape


# ============================================================
# 4. NAÏVE FORECAST
# ============================================================

print("\n==============================")
print("NAÏVE FORECAST")
print("==============================")

naive_results = []

for horizon, target in horizons.items():

    actual = test[target].dropna()

    # Last observed Sales Count from training data
    last_value = train["Sales Count"].iloc[-1]

    predicted = np.full(
        len(actual),
        last_value
    )

    mae, rmse, mape = calculate_metrics(
        actual,
        predicted
    )

    naive_results.append({
        "Model": "Naive",
        "Horizon": horizon,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    })

    print(f"\n{horizon}")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"MAPE : {mape:.2f}%")


# ============================================================
# 5. MOVING AVERAGE FORECAST
# ============================================================

print("\n==============================")
print("MOVING AVERAGE")
print("==============================")

moving_results = []

window = 4

# Last 4 observations from training
last_values = train["Sales Count"].tail(window)

moving_average = last_values.mean()

for horizon, target in horizons.items():

    actual = test[target].dropna()

    predicted = np.full(
        len(actual),
        moving_average
    )

    mae, rmse, mape = calculate_metrics(
        actual,
        predicted
    )

    moving_results.append({
        "Model": "Moving Average",
        "Horizon": horizon,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    })

    print(f"\n{horizon}")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"MAPE : {mape:.2f}%")


# ============================================================
# 6. LINEAR REGRESSION
# ============================================================

print("\n==============================")
print("LINEAR REGRESSION")
print("==============================")

feature_columns = [
    "Sales_Lag_1",
    "Sales_Lag_2",
    "Sales_Lag_4",
    "Sales_Lag_8",
    "Rolling_Mean_4",
    "Rolling_Std_4",
    "Rolling_Max_4",
    "Hour",
    "Day_of_Week",
    "Month",
    "Weekend"
]

linear_results = []

for horizon, target in horizons.items():

    # Remove missing target values
    train_model = train.dropna(
        subset=feature_columns + [target]
    )

    test_model = test.dropna(
        subset=feature_columns + [target]
    )

    X_train = train_model[feature_columns]
    y_train = train_model[target]

    X_test = test_model[feature_columns]
    y_test = test_model[target]

    # Create model
    model = LinearRegression()

    # Train
    model.fit(
        X_train,
        y_train
    )

    # Predict
    predictions = model.predict(X_test)

    # Prevent negative ticket predictions
    predictions = np.maximum(
        predictions,
        0
    )

    mae, rmse, mape = calculate_metrics(
        y_test,
        predictions
    )

    linear_results.append({
        "Model": "Linear Regression",
        "Horizon": horizon,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    })

    print(f"\n{horizon}")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"MAPE : {mape:.2f}%")


# ============================================================
# 7. COMBINE RESULTS
# ============================================================

results = pd.DataFrame(
    naive_results +
    moving_results +
    linear_results
)


# ============================================================
# 8. DISPLAY MODEL COMPARISON
# ============================================================

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

print(
    results.to_string(
        index=False
    )
)


# ============================================================
# 9. SAVE RESULTS
# ============================================================

results.to_csv(
    "baseline_model_results.csv",
    index=False
)

print("\nResults saved as:")
print("baseline_model_results.csv")