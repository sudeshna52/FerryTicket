import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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
# 2. FEATURES
# ============================================================

features = [
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


# ============================================================
# 3. FORECAST HORIZONS
# ============================================================

horizons = {
    "15 Minutes": "Target_15min",
    "30 Minutes": "Target_30min",
    "1 Hour": "Target_1hour",
    "2 Hours": "Target_2hour"
}


# ============================================================
# 4. EVALUATION FUNCTION
# ============================================================

def calculate_metrics(actual, predicted):

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
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
# 5. RANDOM FOREST
# ============================================================

print("\n==============================")
print("RANDOM FOREST REGRESSOR")
print("==============================")

rf_results = []

for horizon, target in horizons.items():

    train_model = train.dropna(
        subset=features + [target]
    )

    test_model = test.dropna(
        subset=features + [target]
    )

    X_train = train_model[features]
    y_train = train_model[target]

    X_test = test_model[features]
    y_test = test_model[target]

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )

    print(f"\nTraining Random Forest for {horizon}...")

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(X_test)

    # Ticket demand cannot be negative
    predictions = np.maximum(
        predictions,
        0
    )

    mae, rmse, mape = calculate_metrics(
        y_test,
        predictions
    )

    rf_results.append({
        "Model": "Random Forest",
        "Horizon": horizon,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape
    })

    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"MAPE : {mape:.2f}%")


# ============================================================
# 6. GRADIENT BOOSTING
# ============================================================

print("\n==============================")
print("GRADIENT BOOSTING REGRESSOR")
print("==============================")

gb_results = []

for horizon, target in horizons.items():

    train_model = train.dropna(
        subset=features + [target]
    )

    test_model = test.dropna(
        subset=features + [target]
    )

    X_train = train_model[features]
    y_train = train_model[target]

    X_test = test_model[features]
    y_test = test_model[target]

    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )

    print(f"\nTraining Gradient Boosting for {horizon}...")

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(X_test)

    # Ticket demand cannot be negative
    predictions = np.maximum(
        predictions,
        0
    )

    mae, rmse, mape = calculate_metrics(
        y_test,
        predictions
    )

    gb_results.append({
        "Model": "Gradient Boosting",
        "Horizon": horizon,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape
    })

    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"MAPE : {mape:.2f}%")


# ============================================================
# 7. COMBINE RESULTS
# ============================================================

results = pd.DataFrame(
    rf_results + gb_results
)


# ============================================================
# 8. DISPLAY MODEL RESULTS
# ============================================================

print("\n==============================")
print("MACHINE LEARNING MODEL RESULTS")
print("==============================")

print(
    results.to_string(index=False)
)


# ============================================================
# 9. SAVE RESULTS
# ============================================================

results.to_csv(
    "ml_model_results.csv",
    index=False
)

print("\nResults saved as:")
print("ml_model_results.csv")